"""Tests for the ttkernel toolchain `ttkernel-ttmetal` (task P17.36).

Bible: IR Strategy (the ttkernel row: ttkernel -> emitc -> C++ -> tt-metal
JIT -> ttsim), Level Rules (TT host programs are never model output),
Toolchain Pins (the joint pin: tt-mlir 0.9.0.dev20260221 at 5f396fd6 with
tt-metal 5280a9cf), Component Interfaces (Toolchain: files -> artifact +
parsed diagnostics), Harness Contract (kernel sources in the build
directory), Agent Rules 6 and 10.

The design these tests fix. The language ttkernel is bound to the Toolchain
"ttkernel-ttmetal" (lassi.toolchains.ttkernel_build). The model writes one
ttkernel MLIR module per kernel, at the kernel path the item's trusted host
program launches with .mlir in place of .cpp. For each module, in sorted
order, the build runs three commands in the build directory, the ones
tt-mlir's own lit tests run at the pin (test/ttmlir/Translate/TTKernel/
ttkernel_noc.mlir:1-2 and ttkernel_compute.mlir:1-2):

    ttmlir-opt <kernel>.mlir -o _ttkernel_emitc/<kernel>.verified.mlir
    ttmlir-opt --convert-ttkernel-to-emitc --mlir-print-debuginfo
        -o _ttkernel_emitc/<kernel>.emitc.mlir <kernel>.mlir
    ttmlir-translate --ttkernel-to-cpp -o <kernel>.cpp _ttkernel_emitc/<kernel>.emitc.mlir

then builds the host program, a harness file, with ttmetal-host's command
line. Every tool error is a Diagnostic (file, line, column, message) at
stage verify (the first command) or lower (the other two), and the host
build's at stage compile. A model file that is not a kernel module (the
host program among them) is refused before anything runs.

No tool runs here: a fake command runner answers each command with
captured-style text (SYNTHETIC, in MLIR's documented diagnostic format, not
captured from the pinned tools) and writes the files a real run would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core.registry import DEFAULT_REGISTRY, RegistryError
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

NAME = "ttkernel-ttmetal"
MODULE = "lassi.toolchains.ttkernel_build"
OPT = "/PLACEHOLDER/toolchains/tt-mlir@0.9.0.dev20260221/build/tt-mlir/bin/ttmlir-opt"
TRANSLATE = "/PLACEHOLDER/toolchains/tt-mlir@0.9.0.dev20260221/build/tt-mlir/bin/ttmlir-translate"
HOST_CLANG = "/usr/bin/clang++-20"
HOST_TREE = "/PLACEHOLDER/toolchains/tt-metal@5280a9cf"
WORK = "_ttkernel_emitc"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
HOST = "copy_example.cpp"
HARNESS = {
    HOST: '#include "lassi_io.h"\nint main() { return 0; }  // SYNTHETIC host program\n',
    "lassi_io.h": "/* SYNTHETIC stand-in for the harness header */\n",
}
KERNEL = "kernels/dataflow/copy.mlir"
KERNEL_TEXT = (
    "// SYNTHETIC ttkernel module written for this test\n"
    "func.func @copy() -> () attributes {ttkernel.thread = #ttkernel.thread<noc>} {\n"
    "  func.return\n"
    "}\n"
)
EMITTED = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {\n  return;\n}\n'


def toolchain_class() -> type:
    """Return the Toolchain class registered as ttkernel-ttmetal; fail clearly while it is missing."""
    try:
        return DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    except RegistryError as error:
        pytest.fail(f"no Toolchain is registered as {NAME!r} ({error}); task P17.36 adds it")


@dataclass
class FakeTools:
    """A CommandRunner standing in for ttmlir-opt, ttmlir-translate, and clang++; it runs nothing.

    `answers` maps a step (verify, lower, translate, host) to (status,
    stderr); a step without one succeeds and writes what the real tool
    writes: the -o file (translate writes `emitted`), or `main` for the host
    build.
    """

    answers: dict[str, tuple[int, str]] = field(default_factory=dict)
    emitted: str = EMITTED
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and return the scripted result, writing the tool's output on success."""
        step = self._step(list(argv))
        self.calls.append((step, list(argv)))
        status, stderr = self.answers.get(step, (0, ""))
        if status == 0:
            out = Path(cwd) / (argv[argv.index("-o") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            text = {"translate": self.emitted, "host": "PLACEHOLDER artifact\n"}.get(step, "module {}\n")
            out.write_bytes(text.encode("utf-8"))
        return CommandResult(status, "", stderr)

    @staticmethod
    def _step(argv: list[str]) -> str:
        """Name the step a command is."""
        if argv[0] == TRANSLATE:
            return "translate"
        if argv[0] == OPT:
            return "lower" if "--convert-ttkernel-to-emitc" in argv else "verify"
        return "host"


def make_toolchain(runner: FakeTools) -> Any:
    """Return the toolchain built as the runner builds it: the host clang, the tree, and the tt-mlir tools."""
    tools = {"TTMLIR_OPT": OPT, "TTMLIR_TRANSLATE": TRANSLATE}
    return toolchain_class()(executable=HOST_CLANG, runner=runner, tree=Path(HOST_TREE), tools=tools)


def build(tmp_path: Path, runner: FakeTools, files: dict[str, str] | None = None) -> Any:
    """Build `files` (default: the one copy kernel) with the harness in a fresh directory; return the result."""
    workdir = tmp_path / "build"
    workdir.mkdir()
    return make_toolchain(runner).build(files or {KERNEL: KERNEL_TEXT}, workdir, harness=HARNESS)


# ---------------------------------------------------------------------------
# Registration and pins


def test_the_package_registers_the_ttkernel_toolchain() -> None:
    cls = toolchain_class()
    assert cls.__module__ == MODULE
    assert cls.name == NAME


def test_it_declares_diagnostics_and_no_host_compute_guard() -> None:
    caps = toolchain_class().capabilities
    assert {"diagnostics", "emits_warnings"} <= caps
    assert "host_compute_guard" not in caps, "the host program is a harness file, never model output"


def test_it_builds_against_the_tt_metal_pin_and_names_the_tt_mlir_tools() -> None:
    cls = toolchain_class()
    assert cls.PIN == "tt-metal"
    assert getattr(cls, "PIN_BIN", None) is None, "the host compiler is the pin's EXECUTABLE"
    assert callable(getattr(cls, "check_tree", None))
    assert dict(cls.TOOL_PINS) == {"tt-mlir": ("TTMLIR_OPT", "TTMLIR_TRANSLATE")}
    pin = pins_module.read_pin("tt-mlir")
    assert pin["TTMLIR_OPT"] == "build/tt-mlir/bin/ttmlir-opt"
    assert pin["TTMLIR_TRANSLATE"] == "build/tt-mlir/bin/ttmlir-translate"


def test_the_commands_are_the_lit_tests_commands() -> None:
    module = __import__(MODULE, fromlist=["x"])
    assert module.verify_command(OPT, KERNEL, f"{WORK}/kernels/dataflow/copy.verified.mlir") == [
        OPT, KERNEL, "-o", f"{WORK}/kernels/dataflow/copy.verified.mlir"]
    assert module.lower_command(OPT, KERNEL, f"{WORK}/kernels/dataflow/copy.emitc.mlir") == [
        OPT, "--convert-ttkernel-to-emitc", "--mlir-print-debuginfo", "-o",
        f"{WORK}/kernels/dataflow/copy.emitc.mlir", KERNEL]
    assert module.translate_command(TRANSLATE, f"{WORK}/kernels/dataflow/copy.emitc.mlir",
                                    "kernels/dataflow/copy.cpp") == [
        TRANSLATE, "--ttkernel-to-cpp", "-o", "kernels/dataflow/copy.cpp", f"{WORK}/kernels/dataflow/copy.emitc.mlir"]
    assert module.kernel_cpp_path("loopback/kernels/copy.mlir") == "loopback/kernels/copy.cpp"


# ---------------------------------------------------------------------------
# Builds


def test_a_build_lowers_each_kernel_and_builds_the_harness_host_program(tmp_path: Path) -> None:
    runner = FakeTools()
    second = "kernels/compute/a.mlir"
    result = build(tmp_path, runner, {KERNEL: KERNEL_TEXT, second: KERNEL_TEXT.replace("noc", "compute")})
    steps = [step for step, _ in runner.calls]
    assert steps == ["verify", "lower", "translate", "verify", "lower", "translate", "host"], (
        "the kernels in sorted order, each verified, lowered, and translated, then one host build")
    assert runner.calls[0][1] == [OPT, second, "-o", f"{WORK}/kernels/compute/a.verified.mlir"]
    host_argv = runner.calls[-1][1]
    assert host_argv[0] == HOST_CLANG and HOST in host_argv, "the host program is the harness's"
    assert "kernels/dataflow/copy.cpp" not in host_argv, "an emitted kernel is placed for the JIT, never compiled"
    workdir = tmp_path / "build"
    assert result.artifact == workdir / OUTPUT
    assert (workdir / "kernels/dataflow/copy.cpp").read_text() == EMITTED, "the kernel C++ lies at its kernel path"
    assert (workdir / KERNEL).read_text() == KERNEL_TEXT
    assert [item for item in result.diagnostics if item.severity == "error"] == []
    assert result.stderr_ref == ATTACHMENT and (workdir / ATTACHMENT).is_file()


def test_a_verifier_error_is_a_parsed_verify_diagnostic_and_stops_that_kernel(tmp_path: Path) -> None:
    # SYNTHETIC stderr in MLIR's diagnostic format: the location, the severity, the message, the source echo, and
    # the caret line, then a note.
    stderr = (
        "kernels/dataflow/copy.mlir:3:5: error: 'ttkernel.cb_push_back' op operand #1 must be 32-bit signless "
        "integer, but got 'index'\n"
        "    ttkernel.cb_push_back(%cb, %n) : (!ttkernel.cb<2, !ttcore.tile<32x32, bf16>>, index) -> ()\n"
        "    ^\n"
        "kernels/dataflow/copy.mlir:3:5: note: see current operation: \"ttkernel.cb_push_back\"(%0, %1)\n"
    )
    runner = FakeTools(answers={"verify": (1, stderr)})
    result = build(tmp_path, runner)
    assert [step for step, _ in runner.calls] == ["verify"], "no lowering, translation, or host build after it"
    assert result.artifact is None
    error, note = result.diagnostics
    assert (error.stage, error.severity, error.file, error.line, error.column) == ("verify", "error", KERNEL, 3, 5)
    assert error.message.startswith("'ttkernel.cb_push_back' op operand #1")
    assert (note.stage, note.severity, note.line) == ("verify", "note", 3)
    assert "ttkernel.cb_push_back(%cb" not in "".join(item.message for item in result.diagnostics), (
        "the source echo is never a diagnostic")


def test_a_lowering_error_is_a_lower_diagnostic_and_unknown_locations_have_no_place(tmp_path: Path) -> None:
    stderr = (
        "kernels/dataflow/copy.mlir:7:3: error: failed to legalize operation 'ttkernel.foo_tile'\n"
        "<unknown>:0: error: conversion to EmitC failed\n"
        "LLVM ERROR: SYNTHETIC fatal error text\n"
    )
    result = build(tmp_path, FakeTools(answers={"lower": (1, stderr)}))
    first, second, third = result.diagnostics
    assert (first.stage, first.file, first.line, first.column) == ("lower", KERNEL, 7, 3)
    assert (second.stage, second.file, second.line, second.column) == ("lower", None, None, None)
    assert second.message == "conversion to EmitC failed"
    assert (third.severity, third.message) == ("error", "SYNTHETIC fatal error text")


def test_a_translation_error_is_a_lower_diagnostic(tmp_path: Path) -> None:
    stderr = "kernels/dataflow/copy.mlir:2:1: error: cannot emit type 'memref<4xi32>'\n"
    runner = FakeTools(answers={"translate": (1, stderr)})
    result = build(tmp_path, runner)
    (error,) = result.diagnostics
    assert (error.stage, error.file, error.line, error.column) == ("lower", KERNEL, 2, 1)
    assert [step for step, _ in runner.calls] == ["verify", "lower", "translate"]


def test_a_failed_tool_with_no_parsed_error_reports_its_exit_status(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(answers={"verify": (139, "Segmentation fault (SYNTHETIC)\n")}))
    (error,) = result.diagnostics
    assert error.severity == "error" and error.code == "exit-status" and error.file == KERNEL
    assert "139" in error.message and "ttmlir-opt" in error.message


def test_errors_of_every_kernel_are_reported_and_no_host_build_runs(tmp_path: Path) -> None:
    stderr_a = "kernels/compute/a.mlir:1:1: error: SYNTHETIC first\n"
    runner = FakeTools(answers={"verify": (1, stderr_a)})
    result = build(tmp_path, runner, {KERNEL: KERNEL_TEXT, "kernels/compute/a.mlir": KERNEL_TEXT})
    assert [step for step, _ in runner.calls] == ["verify", "verify"], "each kernel is checked, then no host build"
    assert len([item for item in result.diagnostics if item.severity == "error"]) == 2


def test_a_module_with_no_kernel_function_is_refused(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(emitted='#include "x.h"\n'))
    (error,) = result.diagnostics
    assert (error.stage, error.code, error.file) == ("lower", "kernel-count", KERNEL)
    assert "ttkernel.thread" in error.message


def test_a_module_with_two_kernel_functions_is_refused(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(emitted=EMITTED + EMITTED))
    (error,) = result.diagnostics
    assert error.code == "kernel-count" and "2" in error.message


def test_a_host_build_error_is_a_compile_diagnostic(tmp_path: Path) -> None:
    stderr = "copy_example.cpp:1:30: error: SYNTHETIC use of undeclared identifier 'x'\n"
    result = build(tmp_path, FakeTools(answers={"host": (1, stderr)}))
    (error,) = [item for item in result.diagnostics if item.severity == "error"]
    assert (error.stage, error.file, error.line) == ("compile", HOST, 1)
    assert result.artifact is None


# ---------------------------------------------------------------------------
# Refusals: the host program is never model output


def test_a_reply_that_writes_the_host_program_is_refused_before_anything_runs(tmp_path: Path) -> None:
    runner = FakeTools()
    result = build(tmp_path, runner, {KERNEL: KERNEL_TEXT, HOST: "int main() { return 0; }\n"})
    assert runner.calls == [], "nothing runs"
    assert result.artifact is None
    (error,) = result.diagnostics
    assert (error.code, error.file, error.severity) == ("bad-path", HOST, "error")
    assert not (tmp_path / "build" / KERNEL).exists(), "nothing is written"


@pytest.mark.parametrize(
    "path",
    ["main.cpp", "kernels/dataflow/copy.cpp", "kernels/helper.h", "copy.mlir", f"{WORK}/kernels/x.mlir"],
)
def test_every_model_file_but_a_kernel_module_is_refused(tmp_path: Path, path: str) -> None:
    runner = FakeTools()
    result = build(tmp_path, runner, {KERNEL: KERNEL_TEXT, path: "// SYNTHETIC\n"})
    assert runner.calls == []
    codes = {item.code for item in result.diagnostics if item.file == path}
    assert codes and codes <= {"bad-path", "not-a-kernel"}, result.diagnostics


def test_a_build_with_no_kernel_module_runs_nothing(tmp_path: Path) -> None:
    runner = FakeTools()
    workdir = tmp_path / "build"
    workdir.mkdir()
    result = make_toolchain(runner).build({}, workdir, harness=HARNESS)
    assert runner.calls == []
    assert [item.code for item in result.diagnostics] == ["no-sources"]


# ---------------------------------------------------------------------------
# The parser alone


def test_parse_mlir_diagnostics_reads_the_mlir_format() -> None:
    module = __import__(MODULE, fromlist=["x"])
    stderr = (
        "./kernels/a.mlir:4:9: warning: SYNTHETIC unused value\n"
        "  %0 = arith.constant 0 : i32\n"
        "        ^\n"
        "kernels/a.mlir:5:1: remark: SYNTHETIC remark\n"
        "error: SYNTHETIC bare error\n"
    )
    found = module.parse_mlir_diagnostics(stderr, "verify")
    assert [(item.severity, item.file, item.line, item.column) for item in found] == [
        ("warning", "kernels/a.mlir", 4, 9), ("note", "kernels/a.mlir", 5, 1), ("error", None, None, None)]
    assert all(item.stage == "verify" for item in found)
