"""Tests for the nvc++ multicore proxy preset, Toolchain "nvcpp-multicore" (DEMO.1).

Bible: Execution Backends, Harness Contract (the bullet "CUDA -> OMP proxy
without a GPU: build a second binary with `nvc++ -mp=multicore` so target
regions run on the host. It checks outputs, never runtime."); Risks And
Questions and the Decision Log, OQ-003 (no NVIDIA host: the compile-only tier
plus the -mp=multicore proxy); Component Interfaces (Toolchain row, the
capability rule).

The contract these tests encode:

- lassi/toolchains/nvcpp.py registers Toolchain "nvcpp-multicore", a
  subclass of NvcppToolchain, beside nvcpp-cc80.
- Its command line is `nvc++ -Wall -O3 -Minfo -mp=multicore -o main
  <sources>`: nvcpp-cc80's LASSI command with -mp=gpu replaced by
  -mp=multicore and no -gpu target, so OpenMP target regions run on the host.
- It shares nvcpp-cc80's pin (PIN "nvhpc" and PIN_BIN) and its stderr parser.
- Its capabilities are nvcpp-cc80's without "openmp_offload", plus
  "openmp_multicore".
- Its docstring cites the Harness Contract bullet and says it is a proxy
  that never appears in faithful recipes and never yields runtime numbers.
- factory() with no arguments builds it, as the stage runner builds a
  toolchain binding, and build() with a fake runner compiles with that
  command.
- The package docstring's preset list names it.

The expected command lines are the task's words and nvcpp-cc80's own
command; the expected diagnostics are nvcpp-cc80's for the same stderr (the
captured fixtures in tests/toolchains/fixtures/, whose hand-derived
diagnostics tests/toolchains/test_diagnostics.py checks). No test here runs a
compiler: build() gets a fake command runner that records its call. No value
in this module is a measurement.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from lassi import toolchains
from lassi.core.capabilities import Component
from lassi.core.interfaces import BuildResult
from lassi.core.registry import DEFAULT_REGISTRY
from lassi.toolchains import CommandResult, nvcpp
from lassi.toolchains.pins import read_pin

NAME = "nvcpp-multicore"
OUTPUT = "main"
ATTACHMENT = "compile.stderr"
# The task's command: nvc++ -Wall -O3 -Minfo -mp=multicore -o main <sources>.
FLAGS = ("-Wall", "-O3", "-Minfo", "-mp=multicore")
CAPABILITIES = frozenset({"openmp_multicore", "emits_warnings", "diagnostics"})

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SOURCES = FIXTURES / "sources"
NVCPP_FIXTURES = sorted(path.stem for path in FIXTURES.glob("nvcpp_*.stderr"))


def preset() -> type:
    """Return the class registered as Toolchain "nvcpp-multicore" (RegistryError while it is not registered)."""
    return DEFAULT_REGISTRY.get("Toolchain", NAME).factory


def fixture_text(name: str) -> str:
    """Return the captured stderr fixture `name`, decoded as UTF-8 with no newline translation."""
    return (FIXTURES / f"{name}.stderr").read_bytes().decode("utf-8")


def scenario_files(name: str) -> dict[str, str]:
    """Return the files scenario `name` compiled (fixtures/sources/<name>/), relative POSIX path -> text."""
    tree = SOURCES / name
    paths = sorted(path for path in tree.rglob("*") if path.is_file())
    return {path.relative_to(tree).as_posix(): path.read_bytes().decode("utf-8") for path in paths}


@dataclass(frozen=True)
class Call:
    """One call the fake runner received."""

    argv: list[str]
    cwd: Path
    timeout_s: float


@dataclass
class FakeRunner:
    """A CommandRunner that records each call and returns canned output; it never starts a process.

    With `creates_output` set it writes the artifact file into the workdir,
    the way a compiler that got far enough would.
    """

    returncode: int = 0
    stderr: str = ""
    creates_output: bool = False
    calls: list[Call] = field(default_factory=list)

    def __call__(self, argv: Sequence[str], cwd: Path, timeout_s: float) -> CommandResult:
        """Record the call, optionally write the artifact, and return the canned CommandResult."""
        self.calls.append(Call(argv=list(argv), cwd=Path(cwd), timeout_s=timeout_s))
        if self.creates_output:
            (Path(cwd) / OUTPUT).write_bytes(b"\x7fELF placeholder artifact written by the fake runner")
        return CommandResult(returncode=self.returncode, stdout="", stderr=self.stderr)


def make_workdir(tmp_path: Path, name: str = "work") -> Path:
    """Return an empty build directory inside the test's temporary directory."""
    workdir = tmp_path / name
    workdir.mkdir()
    return workdir


# ---------------------------------------------------------------------------
# Registration


def test_the_preset_is_registered_in_the_nvcpp_module_as_a_subclass_of_the_adapter() -> None:
    factory = preset()
    assert NAME in DEFAULT_REGISTRY.names("Toolchain")
    assert factory.__module__ == "lassi.toolchains.nvcpp"
    assert getattr(nvcpp, factory.__name__) is factory
    assert issubclass(factory, nvcpp.NvcppToolchain)
    assert factory is not nvcpp.NvcppCc80 and not issubclass(factory, nvcpp.NvcppCc80)
    assert factory.name == NAME


def test_capabilities_replace_openmp_offload_with_openmp_multicore() -> None:
    entry = DEFAULT_REGISTRY.get("Toolchain", NAME)
    assert entry.capabilities == CAPABILITIES
    assert entry.factory.capabilities == CAPABILITIES
    assert "openmp_offload" not in entry.capabilities
    assert CAPABILITIES == (nvcpp.NvcppCc80.capabilities - {"openmp_offload"}) | {"openmp_multicore"}
    # The GPU preset keeps its own capabilities.
    assert DEFAULT_REGISTRY.get("Toolchain", "nvcpp-cc80").capabilities == nvcpp.NvcppCc80.capabilities
    assert "openmp_offload" in nvcpp.NvcppCc80.capabilities


def test_the_preset_shares_the_nvcpp_cc80_pin() -> None:
    factory = preset()
    assert factory.PIN == nvcpp.NvcppCc80.PIN == "nvhpc"
    assert factory.PIN_BIN == nvcpp.NvcppCc80.PIN_BIN
    pin = read_pin(factory.PIN)
    assert factory.PIN_BIN.format_map(pin).endswith("/nvc++")


# ---------------------------------------------------------------------------
# The command line


def test_command_line_is_exact() -> None:
    argv = preset()().command(["main.cpp", "util/io.c"])
    assert type(argv) is list
    assert argv == ["nvc++", "-Wall", "-O3", "-Minfo", "-mp=multicore", "-o", "main", "main.cpp", "util/io.c"]
    assert not any(word.startswith("-gpu") for word in argv), argv


def test_command_is_the_nvcpp_cc80_command_with_the_offload_flags_replaced() -> None:
    gpu = nvcpp.NvcppCc80().command(["main.cpp"])
    expected = ["-mp=multicore" if word == "-mp=gpu" else word for word in gpu if not word.startswith("-gpu=")]
    assert preset()().command(["main.cpp"]) == expected


def test_executable_setting_replaces_only_the_first_word() -> None:
    argv = preset()(executable="/opt/nvhpc-pin/bin/nvc++").command(["main.cpp"])
    assert argv == ["/opt/nvhpc-pin/bin/nvc++", *FLAGS, "-o", OUTPUT, "main.cpp"]


# ---------------------------------------------------------------------------
# The parser


@pytest.mark.parametrize("name", NVCPP_FIXTURES)
def test_the_parser_is_nvcpp_cc80s(name: str) -> None:
    stderr, files = fixture_text(name), scenario_files(name)
    expected = nvcpp.parse_diagnostics(stderr, files)
    assert nvcpp.NvcppCc80().parse(stderr, files) == expected
    assert preset()().parse(stderr, files) == expected


def test_the_fixtures_include_an_error_and_a_warning() -> None:
    # Guards the parametrized parser test above against an empty or thinned fixture set.
    assert {"nvcpp_edg_error", "nvcpp_edg_warning", "nvcpp_minfo_clean"} <= set(NVCPP_FIXTURES)


# ---------------------------------------------------------------------------
# Construction and build with a fake runner


def test_factory_with_no_arguments_builds_the_preset() -> None:
    # Toolchain bindings carry no config, so the runner builds them as factory().
    factory = preset()
    tool = DEFAULT_REGISTRY.get("Toolchain", NAME).factory()
    assert type(tool) is factory
    assert isinstance(tool, Component)
    assert tool.name == NAME
    assert tool.command(["x.c"]) == ["nvc++", *FLAGS, "-o", OUTPUT, "x.c"]
    assert list(inspect.signature(factory.build).parameters) == ["self", "files", "workdir", "harness"]


def test_factory_with_no_arguments_builds_through_subprocess_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # runner=None means subprocess_runner; a fake stands in for it here, so no compiler starts.
    fake = FakeRunner(creates_output=True)
    monkeypatch.setattr("lassi.toolchains._base.subprocess_runner", fake)
    workdir = make_workdir(tmp_path)
    result = DEFAULT_REGISTRY.get("Toolchain", NAME).factory().build({"main.cpp": "int x;\n"}, workdir)
    assert [(call.argv, call.cwd, call.timeout_s) for call in fake.calls] == [
        (["nvc++", *FLAGS, "-o", OUTPUT, "main.cpp"], workdir, 600.0)
    ]
    assert result == BuildResult(artifact=workdir / OUTPUT, diagnostics=[], stderr_ref=ATTACHMENT)


def test_build_passes_only_sources_in_sorted_order(tmp_path: Path) -> None:
    names = ["z.cu", "b.cpp", "g.h", "c.cc", "d.cxx", "e.c", "h.hpp", "notes.txt", "src/k/a.cpp", "reference.h"]
    runner = FakeRunner(creates_output=True)
    workdir = make_workdir(tmp_path)
    preset()(runner=runner).build({name: "int x;\n" for name in names}, workdir)
    sources = ["b.cpp", "c.cc", "d.cxx", "e.c", "src/k/a.cpp"]
    assert [(call.argv, call.cwd, call.timeout_s) for call in runner.calls] == [
        (["nvc++", *FLAGS, "-o", OUTPUT, *sources], workdir, 600.0)
    ]


def test_build_reports_the_same_diagnostics_as_nvcpp_cc80(tmp_path: Path) -> None:
    stderr, files = fixture_text("nvcpp_edg_error"), scenario_files("nvcpp_edg_error")
    gpu = nvcpp.NvcppCc80(runner=FakeRunner(returncode=1, stderr=stderr)).build(files, make_workdir(tmp_path, "gpu"))
    proxy_dir = make_workdir(tmp_path, "proxy")
    proxy = preset()(runner=FakeRunner(returncode=1, stderr=stderr)).build(files, proxy_dir)
    assert gpu.diagnostics  # the fixture is a compile error
    assert proxy == BuildResult(artifact=None, diagnostics=gpu.diagnostics, stderr_ref=ATTACHMENT)
    assert (proxy_dir / ATTACHMENT).read_bytes() == stderr.encode("utf-8")


# ---------------------------------------------------------------------------
# Documentation


def test_docstring_cites_the_harness_contract_and_says_what_the_proxy_never_does() -> None:
    doc = inspect.getdoc(preset()) or ""
    lowered = doc.lower()
    assert "Harness Contract" in doc, doc
    assert "-mp=multicore" in doc, doc
    assert "proxy" in lowered, doc
    assert "faithful" in lowered, doc  # never appears in faithful recipes
    assert "runtime" in lowered, doc  # never yields runtime numbers
    assert "never" in lowered, doc


def test_the_cited_harness_contract_bullet_is_in_the_bible() -> None:
    bible = (Path(__file__).resolve().parents[2] / "docs" / "BIBLE.md").read_text(encoding="utf-8").splitlines()
    start = bible.index("### Harness Contract")
    end = next(index for index in range(start + 1, len(bible)) if bible[index].startswith("#"))
    bullets = [line for line in bible[start:end] if line.startswith("- CUDA -> OMP proxy without a GPU")]
    assert len(bullets) == 1, bullets
    assert "nvc++ -mp=multicore" in bullets[0] and "never runtime" in bullets[0]


def test_the_package_docstring_lists_the_preset() -> None:
    doc = toolchains.__doc__ or ""
    assert f'"{NAME}"' in doc, doc
    assert f"nvcpp.{preset().__name__}" in doc, doc
