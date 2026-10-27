"""Tests for the lowering toolchain `cgeist-affine` (task P17.41): the pinned cgeist on a compute-only C unit.

Bible: IR Strategy (the cpu-mlir row: cgeist raises C to affine, scf, and
memref), Language Frontends (C through cgeist), Toolchain Pins (MLIR stacks:
Polygeist 77c04bb2 on LLVM 26eb4285; toolchains/polygeist.pin), Agent Rules
6 and 10. plans/spikes/mlir-issues.md, Half 2: cgeist aborts on the items'
whole sources; on a compute-only unit it lowers one function by its C name;
and the pinned polygeist-opt has no --raise-affine-to-linalg, so the level
is affine.

The design these tests fix. The Toolchain cgeist-affine declares
lowers_source (lassi.core.capabilities) and nothing a stage needs to build a
language, so it is bound only by a recipe's source_mlir section. It declares
PIN polygeist and PIN_BIN {CGEIST}, so the runner builds it as every pinned
toolchain: the executable under the toolchains root, the compile sandbox,
and `cgeist --version` against the pin's EXPECT_VERSION. lower(file, text,
function, workdir) writes the unit and runs, in the compile sandbox,

    cgeist <file> -S --raise-scf-to-affine --function=<function> -o _cgeist/<stem>.affine.mlir

(tools/cgeist/driver.cc:146-147, :175-176, :211-213 at 77c04bb2), and
returns the module, or none with a parsed error; it never raises for the
tool's failure. No tool runs here: a fake command runner answers each
command with SYNTHETIC text and writes the file a real run would.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import lassi.toolchains  # noqa: F401  (importing the package registers every toolchain preset)
from lassi.core import runner as runner_module
from lassi.core.capabilities import LOWERS_SOURCE
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.core.runner import RunError, build_toolchain
from lassi.toolchains import CommandResult
from lassi.toolchains import pins as pins_module

NAME = "cgeist-affine"
EXE = "/PLACEHOLDER/toolchains/polygeist@77c04bb2/build/bin/cgeist"
UNIT = "/* SYNTHETIC unit */\nvoid lassi_scale(const float *x, float *y, long n) {}\n"
MODULE = "module {\n  func.func @lassi_scale() {\n    affine.for %i = 0 to 4 {\n    }\n    return\n  }\n}\n"
ASSERTION = (
    "cgeist: /PLACEHOLDER/tools/cgeist/Lib/ValueCategory.cc:154: void ValueCategory::store(...): Assertion "
    "`toStore.isReference' failed.\nPLEASE submit a bug report\n"
)


@dataclass
class Fake:
    """What the fake runner answers: an exit status, stderr, and whether it writes the -o file; and its calls."""

    status: int = 0
    stderr: str = ""
    writes: bool = True
    calls: list[tuple[list[str], Path]] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call and write the module at -o when it is to."""
        self.calls.append((list(argv), Path(cwd)))
        if self.writes:
            out = Path(cwd) / argv[argv.index("-o") + 1]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(MODULE.encode("ascii"))
        return CommandResult(self.status, "", self.stderr)


def toolchain(fake: Fake, timeout_s: float = 300.0) -> Any:
    """Return cgeist-affine on the fake runner."""
    return DEFAULT_REGISTRY.get("Toolchain", NAME).factory(executable=EXE, runner=fake, timeout_s=timeout_s)


def test_the_toolchain_is_registered_and_pinned() -> None:
    entry = DEFAULT_REGISTRY.get("Toolchain", NAME)
    assert entry.capabilities == frozenset({LOWERS_SOURCE}), "never a language's toolchain: no diagnostics capability"
    cls = entry.factory
    assert (cls.PIN, cls.PIN_BIN, cls.LEVEL) == ("polygeist", "{CGEIST}", "affine")


def test_lower_runs_the_pinned_command_and_returns_the_module(tmp_path: Path) -> None:
    fake = Fake()
    lowering = toolchain(fake).lower("scale.c", UNIT, "lassi_scale", tmp_path)
    assert fake.calls == [([EXE, "scale.c", "-S", "--raise-scf-to-affine", "--function=lassi_scale", "-o",
                            "_cgeist/scale.affine.mlir"], tmp_path)]
    assert (tmp_path / "scale.c").read_bytes() == UNIT.encode("ascii")
    assert lowering.module == MODULE and lowering.level == "affine" and lowering.diagnostics == []


def test_an_assertion_is_an_error_naming_the_status_and_the_line(tmp_path: Path) -> None:
    lowering = toolchain(Fake(status=134, stderr=ASSERTION, writes=False)).lower("scale.c", UNIT, "f", tmp_path)
    assert lowering.module is None
    errors = [item for item in lowering.diagnostics if item.severity == "error"]
    assert len(errors) == 1 and errors[0].stage == "lower" and errors[0].code == "exit-status"
    assert "134" in errors[0].message and "toStore.isReference" in errors[0].message
    assert errors[0].message.isascii()


def test_a_clang_error_is_parsed_with_its_place(tmp_path: Path) -> None:
    stderr = "scale.c:2:6: error: unknown type name 'flot'\n    flot x;\n    ^\n"
    lowering = toolchain(Fake(status=1, stderr=stderr, writes=False)).lower("scale.c", UNIT, "f", tmp_path)
    assert lowering.module is None
    found = lowering.diagnostics[0]
    assert (found.file, found.line, found.column, found.message) == ("scale.c", 2, 6, "unknown type name 'flot'")


@pytest.mark.parametrize(("status", "writes", "words"), [(-1, False, "timed out"), (0, False, "no module")])
def test_a_timeout_or_a_missing_module_is_an_error(tmp_path: Path, status: int, writes: bool, words: str) -> None:
    lowering = toolchain(Fake(status=status, writes=writes)).lower("scale.c", UNIT, "f", tmp_path)
    assert lowering.module is None and words in lowering.diagnostics[-1].message


@pytest.mark.parametrize("file", ["../scale.c", "scale.cpp.txt", "-scale.c", "_cgeist/scale.c"])
def test_a_bad_unit_name_is_refused_before_anything_runs(tmp_path: Path, file: str) -> None:
    fake = Fake()
    lowering = toolchain(fake).lower(file, UNIT, "f", tmp_path)
    assert fake.calls == [] and lowering.module is None and lowering.diagnostics[0].code == "bad-path"


def test_a_bad_function_name_is_refused_before_anything_runs(tmp_path: Path) -> None:
    fake = Fake()
    lowering = toolchain(fake).lower("scale.c", UNIT, "f; rm", tmp_path)
    assert fake.calls == [] and lowering.module is None and "function" in lowering.diagnostics[0].message


def test_build_refuses_with_a_diagnostic(tmp_path: Path) -> None:
    result = toolchain(Fake()).build({"scale.c": UNIT}, tmp_path)
    assert result.artifact is None and "source_mlir" in result.diagnostics[0].message


# ---------------------------------------------------------------------------
# build_toolchain with the pin (the runner's route)


@dataclass
class Versions:
    """What the fake compile runner saw; it answers --version with `banner`."""

    banner: str
    calls: list[list[str]] = field(default_factory=list)


def fake_compile_runner(log: Versions) -> type:
    """Return a stand-in for SandboxedCompileRunner that answers --version only."""

    class FakeCompileRunner:
        """Answers --version with the log's banner; any other command fails the test."""

        def __init__(self, **keywords: Any) -> None:
            self.keywords = keywords

        def spec(self, build: Path) -> None:
            """Accept any layout."""

        def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
            log.calls.append(list(argv))
            assert list(argv[1:]) == ["--version"], argv
            return CommandResult(0, log.banner, "")

    return FakeCompileRunner


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a stand-in toolchains root holding the pinned cgeist's path, with TMPDIR and HOME set."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    for name in ("home", "compile-tmp"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "compile-tmp"))
    pin = pins_module.read_pin("polygeist")
    found = tmp_path / "toolchains-root"
    exe = found / pin["PREFIX_NAME"] / pin["CGEIST"]
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    return found


def test_build_toolchain_checks_the_pinned_version_and_records_the_pin(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = pins_module.read_pin("polygeist")
    log = Versions(banner=f"PLACEHOLDER banner of a fake runner\n  {pin['EXPECT_VERSION']}\n")
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", fake_compile_runner(log))
    built = build_toolchain(NAME, root)
    exe = root.resolve() / pin["PREFIX_NAME"] / pin["CGEIST"]
    assert log.calls == [[str(exe), "--version"]]
    assert built.executable == str(exe) and set(built.pins) == {"polygeist"}
    assert built.toolchain.executable == str(exe)


def test_build_toolchain_refuses_another_version(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "SandboxedCompileRunner", fake_compile_runner(Versions("LLVM version 19\n")))
    with pytest.raises(RunError, match="EXPECT_VERSION|version"):
        build_toolchain(NAME, root)
