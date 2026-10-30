"""Tests for the compile-only TT-NN toolchain ttir-ttnn-compile, the language ttnn (task P17.57).

Bible: IR Strategy (Level Rules; the gather and scatter routes), Toolchain
Pins (tt-mlir 0.9.0.dev20260221 at 5f396fd6), Component Interfaces
(Toolchain: files -> artifact + parsed diagnostics), Agent Rules 3, 6, and
11. Evidence: plans/spikes/scatter-gather-workarounds.md, "Compile probe at
the pin (2026-10-29)".

The design these tests fix. The language ttnn is route A of the spike: the
model writes one TTIR module, <m>.mlir, and the Toolchain
"ttir-ttnn-compile" (lassi.toolchains.ttir_compile.TtirTtnnCompile) runs, in
the build directory:

    ttmlir-opt <m>.mlir -o _ttnn/<m>.verified.mlir
    ttmlir-opt --ttir-to-ttnn-backend-pipeline <m>.mlir -o _ttnn/<m>.ttnn.mlir
    ttmlir-translate --ttnn-to-flatbuffer _ttnn/<m>.ttnn.mlir -o _ttnn/<m>.ttnn

The artifact is the flatbuffer when every step passed, the lowered module
holds a ttnn op that computes (not only device, layout, free, or creation
ops), and the flatbuffer carries TTNN after its size prefix (bytes 8 to 12).
The module is TTIR: a ttir op, no d2m op, and no ttnn op (the compiler
writes those). Nothing here declares that an executor may run the output.

No tool runs here: a fake command runner answers each command and writes
the files a real run would. The warning and the lowered excerpts are copied
from the probe's outputs (exploratory runs, rx 20261029-204757-exec-d2a1
and 20261029-204857-exec-9cd1, on hand-written SYNTHETIC modules that are
no bench item's); the error texts are SYNTHETIC, in MLIR's documented
diagnostic format and glibc's assertion format.
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

NAME = "ttir-ttnn-compile"
OPT = "/PLACEHOLDER/toolchains/tt-mlir@0.9.0.dev20260221/build/tt-mlir/bin/ttmlir-opt"
TRANSLATE = "/PLACEHOLDER/toolchains/tt-mlir@0.9.0.dev20260221/build/tt-mlir/bin/ttmlir-translate"
WORK = "_ttnn"
ATTACHMENT = "compile.stderr"
MODULE = "probe.mlir"
# A SYNTHETIC TTIR module written for these tests; it is no bench item's.
TTIR_TEXT = (
    "// SYNTHETIC TTIR module written for this test\n"
    "module {\n"
    "  func.func @probe(%x: tensor<2048xsi32>) -> tensor<2048xsi32> {\n"
    '    %0 = "ttir.cumsum"(%x) <{dim = 0 : i64}> : (tensor<2048xsi32>) -> tensor<2048xsi32>\n'
    "    return %0 : tensor<2048xsi32>\n"
    "  }\n"
    "}\n"
)
# The warning the pinned TT-NN pipeline printed on every probe module, at the module's first op, with its echo.
WARNING = (
    "probe.mlir:2:1: warning: Empty argument type map provided. Skipping argument type population. This may affect "
    "subsequent compile steps\n"
    "module {\n"
    "^\n"
)
# An excerpt of the lowered module of the probe's sort.mlir (rx 20261029-204857-exec-9cd1), layouts first.
LOWERED = (
    "#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x64x!ttcore.tile<32x32, si32>, #dram>, "
    "<interleaved>>\n"
    '%0 = "ttnn.typecast"(%arg0) <{dtype = #ttcore.supportedDataTypes<u16>}> : (tensor<2048xsi32, #ttnn_layout>) -> '
    "tensor<2048xui16, #ttnn_layout1>\n"
    '"ttnn.deallocate"(%arg0) <{force = false}> : (tensor<2048xsi32, #ttnn_layout>) -> ()\n'
    '%values, %indices = "ttnn.sort"(%0) <{descending = false, dim = -1 : si8, stable = true}> : (tensor<2048xui16, '
    "#ttnn_layout1>) -> (tensor<2048xui16, #ttnn_layout1>, tensor<2048xui16, #ttnn_layout1>)\n"
)
# The probe's identity module lowered to a function with no ttnn op at all (rx 20261029-204857-exec-9cd1).
IDENTITY = (
    'func.func @identity(%arg0: tensor<2048xsi32, #ttnn_layout>) -> tensor<2048xsi32, #ttnn_layout> attributes '
    '{tt.function_type = "forward_device"} {\n'
    "  return %arg0 : tensor<2048xsi32, #ttnn_layout>\n"
)
# The first 12 bytes of the probe's cumsum.ttnn (rx 20261029-204857-exec-9cd1): the size 7100, the root offset 28, TTNN.
FLATBUFFER = b"\xbc\x1b\x00\x00\x1c\x00\x00\x00TTNN" + bytes(16)


@dataclass
class FakeTools:
    """A CommandRunner standing in for ttmlir-opt and ttmlir-translate on the TT-NN route; it runs nothing.

    `answers` maps a step (verify, lower, translate) to (status, stderr); a
    step without one succeeds, prints `warning` for lower, and writes its
    -o file: `printed` for verify, `lowered` for lower, and `flatbuffer`
    for translate.
    """

    printed: str = TTIR_TEXT
    lowered: str = LOWERED
    flatbuffer: bytes = FLATBUFFER
    warning: str = WARNING
    answers: dict[str, tuple[int, str]] = field(default_factory=dict)
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and return the scripted result, writing the tool's output on success."""
        step = self._step(list(argv))
        self.calls.append((step, list(argv)))
        status, stderr = self.answers.get(step, (0, self.warning if step == "lower" else ""))
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
        return "lower" if "--ttir-to-ttnn-backend-pipeline" in argv else "verify"


def toolchain(runner: FakeTools) -> Any:
    """Return the toolchain built as the runner builds it: the pinned ttmlir-opt and ttmlir-translate."""
    cls = DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    return cls(executable=OPT, runner=runner, tools={"TTMLIR_TRANSLATE": TRANSLATE})


def build(tmp_path: Path, runner: FakeTools, files: dict[str, str] | None = None) -> Any:
    """Build `files` (default: the one SYNTHETIC module) in a fresh directory; return the BuildResult."""
    workdir = tmp_path / "build"
    workdir.mkdir(parents=True)
    return toolchain(runner).build(files or {MODULE: TTIR_TEXT}, workdir)


def errors(result: Any) -> list[Any]:
    """Return the error diagnostics of a build."""
    return [item for item in result.diagnostics if item.severity == "error"]


# ---------------------------------------------------------------------------
# Registration, pins, and commands


def test_it_is_registered_compile_only_beside_the_ttmetal_toolchains() -> None:
    cls = DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    assert cls.__module__ == "lassi.toolchains.ttir_compile" and cls.name == NAME
    assert cls is lassi.toolchains.TtirTtnnCompile
    assert cls.capabilities == frozenset({"diagnostics", "emits_warnings"}), "nothing an executor reads to run it"
    assert not hasattr(cls, "check_tree") and cls.SOURCE_SUFFIXES == (".mlir",)
    assert cls.BACKEND is ttir_compile.TTNN and cls.LANGUAGE == "TTIR"
    ttir = DEFAULT_REGISTRY.get("Toolchain", "ttir-ttmetal-compile").factory
    assert ttir.BACKEND is ttir_compile.TTMETAL, "the TT-Metal route is unchanged"


def test_it_runs_the_pinned_tt_mlir_tools() -> None:
    cls = DEFAULT_REGISTRY.get("Toolchain", NAME).factory
    pin = pins_module.read_pin("tt-mlir")
    assert cls.PIN == "tt-mlir" and cls.PIN_BIN.format_map(pin) == pin["TTMLIR_OPT"]
    assert dict(cls.TOOL_PINS) == {"tt-mlir": ("TTMLIR_TRANSLATE",)}
    with pytest.raises(ValueError, match="tool pin"):
        cls.check_tool_pin("tt-metal", Path("."), pin)


def test_the_route_is_the_probes() -> None:
    backend = ttir_compile.TTNN
    assert (backend.pipeline, backend.translation) == ("--ttir-to-ttnn-backend-pipeline", "--ttnn-to-flatbuffer")
    assert (backend.work_dir, backend.lowered_suffix, backend.flatbuffer_suffix) == (WORK, ".ttnn.mlir", ".ttnn")
    assert backend.identifier == b"TTNN" and FLATBUFFER[ttir_compile.IDENTIFIER_SPAN] == b"TTNN"
    assert "system-desc" not in backend.pipeline, "the mock descriptor: compile evidence only"


def test_a_module_that_compiles_runs_the_three_commands_and_gives_the_flatbuffer(tmp_path: Path) -> None:
    runner = FakeTools()
    result = build(tmp_path, runner)
    assert [argv for _, argv in runner.calls] == [
        [OPT, MODULE, "-o", f"{WORK}/probe.verified.mlir"],
        [OPT, "--ttir-to-ttnn-backend-pipeline", MODULE, "-o", f"{WORK}/probe.ttnn.mlir"],
        [TRANSLATE, "--ttnn-to-flatbuffer", f"{WORK}/probe.ttnn.mlir", "-o", f"{WORK}/probe.ttnn"],
    ]
    workdir = tmp_path / "build"
    assert result.artifact == workdir / WORK / "probe.ttnn"
    assert errors(result) == [] and result.stderr_ref == ATTACHMENT
    (warning,) = result.diagnostics
    assert (warning.severity, warning.stage, warning.file, warning.line) == ("warning", "lower", MODULE, 2)
    assert "Empty argument type map" in warning.message
    log = (workdir / ATTACHMENT).read_text(encoding="utf-8")
    assert log.count("$ ") == 3 and "--ttnn-to-flatbuffer" in log


# ---------------------------------------------------------------------------
# Failures


def test_a_verify_error_is_parsed_and_stops_the_build(tmp_path: Path) -> None:
    stderr = "probe.mlir:4:10: error: 'ttir.cumsum' op requires attribute 'dim'\n"
    runner = FakeTools(answers={"verify": (1, stderr)})
    result = build(tmp_path, runner)
    assert [step for step, _ in runner.calls] == ["verify"] and result.artifact is None
    (found,) = errors(result)
    assert (found.stage, found.file, found.line, found.column) == ("verify", MODULE, 4, 10)


def test_a_lowering_error_and_an_abort_are_errors(tmp_path: Path) -> None:
    stderr = "probe.mlir:4:10: error: failed to legalize operation 'ttir.cumsum' that was explicitly marked illegal\n"
    result = build(tmp_path, FakeTools(answers={"lower": (1, stderr)}))
    (found,) = errors(result)
    assert (found.stage, found.file, found.line) == ("lower", MODULE, 4)
    aborted = "ttmlir-opt: /src/lib/X.cpp:10: void f(): Assertion `ok && \"bad shape\"' failed.\n"
    result = build(tmp_path / "abort", FakeTools(answers={"lower": (134, aborted)}))
    (found,) = errors(result)
    assert found.code == "abort" and "bad shape" in found.message and "exit status 134" in found.message


def test_a_translate_error_in_the_lowered_module_loses_its_place(tmp_path: Path) -> None:
    stderr = f"{WORK}/probe.ttnn.mlir:12:3: error: unsupported op in flatbuffer translation\n"
    result = build(tmp_path, FakeTools(answers={"translate": (1, stderr)}))
    (found,) = errors(result)
    assert found.file is None and f"at {WORK}/probe.ttnn.mlir:12:3" in found.message
    assert "not in your file" in found.message


def test_a_module_with_no_ttnn_op_that_computes_fails(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(lowered=IDENTITY))
    assert result.artifact is None
    (found,) = errors(result)
    assert found.code == "no-device-program" and "no ttnn op at all" in found.message
    glue = '%0 = "ttnn.zeros"() : () -> tensor<4096xsi32>\n"ttnn.deallocate"(%0) : () -> ()\n'
    glue += '%1 = "ttnn.to_layout"(%0) <{layout = #ttnn.layout<row_major>}> : () -> ()\n'
    result = build(tmp_path / "glue", FakeTools(lowered=glue))
    (found,) = errors(result)
    assert found.code == "no-device-program"
    assert "ttnn.deallocate, ttnn.to_layout, ttnn.zeros" in found.message


@pytest.mark.parametrize("head", [
    b"\xbc\x1b\x00\x00\x1c\x00\x00\x00TTM0",  # a TT-Metal flatbuffer: the other route
    b"\x1c\x00\x00\x00TTNN\x00\x00\x00\x00",  # TTNN at bytes 4 to 8: not this pin's size-prefixed form
    b"not a flatbuffer",
])
def test_a_file_that_is_no_ttnn_flatbuffer_fails(tmp_path: Path, head: bytes) -> None:
    result = build(tmp_path, FakeTools(flatbuffer=head + bytes(16)))
    assert result.artifact is None
    (found,) = errors(result)
    assert found.code == "no-artifact" and "TT-NN flatbuffer" in found.message and "TTNN" in found.message


# ---------------------------------------------------------------------------
# The level rule and the files


@pytest.mark.parametrize(("printed", "named"), [
    (TTIR_TEXT + '%1 = "ttnn.sort"(%0) : (tensor<2048xui16>) -> tensor<2048xui16>\n', "ttnn.sort"),
    (TTIR_TEXT + "%z = d2m.empty() : tensor<1x1xf32>\n", "d2m.empty"),
])
def test_the_module_is_ttir_and_never_the_compilers_lower_levels(tmp_path: Path, printed: str, named: str) -> None:
    runner = FakeTools(printed=printed)
    result = build(tmp_path, runner)
    assert [step for step, _ in runner.calls] == ["verify"]
    (found,) = errors(result)
    assert (found.code, found.stage) == ("wrong-level", "verify") and named in found.message
    assert "lowers it to TTNN itself" in found.message


def test_a_module_with_no_ttir_op_is_the_wrong_level(tmp_path: Path) -> None:
    result = build(tmp_path, FakeTools(printed="module {\n  func.func @f() { return }\n}\n"))
    assert errors(result)[0].code == "wrong-level" and "no TTIR op" in errors(result)[0].message


def test_ttnn_attributes_and_types_are_not_ttnn_ops() -> None:
    assert ttir_compile.ttnn_ops(LOWERED) == ["typecast", "deallocate", "sort"]
    text = '#l = #ttnn.ttnn_layout<x>\n%d = "ttnn.get_device"() : () -> !ttnn.device\n{ttnn.hoisted = true}\n'
    assert ttir_compile.ttnn_ops(text) == ["get_device"]


def test_other_files_a_second_module_and_the_work_dir_are_refused_before_anything_runs(tmp_path: Path) -> None:
    runner = FakeTools()
    result = build(tmp_path, runner, files={MODULE: TTIR_TEXT, "notes.txt": "x\n", "other.mlir": TTIR_TEXT})
    assert runner.calls == [] and result.artifact is None
    assert sorted(item.code for item in errors(result)) == ["module-count", "not-a-module"]
    result = build(tmp_path / "work", runner, files={f"{WORK}/probe.mlir": TTIR_TEXT})
    assert runner.calls == [] and errors(result)[0].code == "bad-path" and WORK in errors(result)[0].message


def test_no_module_is_a_no_sources_error(tmp_path: Path) -> None:
    runner = FakeTools()
    workdir = tmp_path / "build"
    workdir.mkdir()
    result = toolchain(runner).build({}, workdir)
    assert runner.calls == [] and errors(result)[0].code == "no-sources" and NAME in errors(result)[0].message
