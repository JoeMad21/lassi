"""Tests for tools/ttsim_smoke.py, the ttsim smoke driver (task P4.11), with fakes.

Bible: Execution Backends (the ttsim row), ttsim Facts, Result Record
(provenance), Agent Rules 1, 2, 6, 7, and 10. Plan: plans/p4-ttsim.md,
P4.11; the examples' own checks are in plans/spikes/p4-ttsim-runtime.md
("The examples' own checks").

The interface these tests fix:

- `main(argv) -> int`, argv without the program name: one example, one of
  the six P4.9 ran (add_2_integers_in_riscv, loopback, eltwise_binary,
  eltwise_sfpu, matmul_single_core, matmul_multi_core). Exit status 0 on a
  pass, 1 on a fail, and 2 on a usage error or a refusal (an unknown
  example, $LASSI_RUNS_ROOT or $LASSI_TOOLCHAINS unset, an install that is
  missing); a usage error may come as argparse's SystemExit(2).
- The example is read from the pinned tree at run time (its host program
  and each kernel it names after OVERRIDE_KERNEL_PREFIX, relative to
  tt_metal/programming_examples, and, for the matmuls, Matmul::Common's
  bmm_op.hpp at the path the program includes it by), built with
  lassi.core.runner.build_toolchain("ttmetal-host", ...) into a new
  directory under $LASSI_RUNS_ROOT/ttsim-smoke/<id>/, and run through
  lassi.executors.ttsim.TtsimExecutor with 16 CPUs.
- It passes only when the build gave an artifact, the run exited 0 with no
  hang, sim_ub is False, sim_gap is None, no jit-stage error came back, and
  stdout holds the example's own success line ("Success: Result is 21" for
  the gate's example, "Test Passed" for the others), and for the matmuls
  "Metalium vs Golden -- PCC = <value>" with a value above 0.97.
- It writes <id>/summary.json, plain ASCII, with commit, dirty, pins
  (tt-metal, and ttsim with its VERSION and SHA256), device
  (TtsimExecutor.device()), host, date (UTC, ISO 8601), argv, the exit
  status, hang, sim_ub, sim_gap, findings, the success line found, the PCC
  when printed, wall_s labeled "simulator wall time, exploratory, sizing
  only", and the verdict, and prints the summary's path.

The fakes: the toolchains root is ttsim_fakes.make_install's PLACEHOLDER
install, with SYNTHETIC stand-ins for each example's host program, kernel,
and header in the tree (made up here, never compiled; no tt-metal text);
build_toolchain is replaced by one whose toolchain writes a PLACEHOLDER
artifact and records what it was given; and Sandbox.run is replaced, so the
real executor runs nothing and returns canned SYNTHETIC output. git runs for
real for the commit and dirty flag. No value in this module is a
measurement.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult, Limits
from lassi.core.record import Diagnostic
from lassi.core.runner import BuiltToolchain, RunError
from lassi.executors import sandbox as sandbox_module
from lassi.toolchains.pins import read_pin

REPO = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO / "tools"
TOOL_MODULE = "ttsim_smoke"
EXECUTOR_TESTS = REPO / "tests" / "executors"
if str(EXECUTOR_TESTS) not in sys.path:
    sys.path.insert(0, str(EXECUTOR_TESTS))
from ttsim_fakes import FakeInstall, capture_text, make_install, same_path  # noqa: E402

EXAMPLES_DIR = "tt_metal/programming_examples"
# Each example's directory under EXAMPLES_DIR, as the pinned tree lays them out (the ttmetal-host remote test).
EXAMPLES = {
    "add_2_integers_in_riscv": "add_2_integers_in_riscv",
    "loopback": "loopback",
    "eltwise_binary": "eltwise_binary",
    "eltwise_sfpu": "eltwise_sfpu",
    "matmul_single_core": "matmul/matmul_single_core",
    "matmul_multi_core": "matmul/matmul_multi_core",
}
MATMULS = ("matmul_single_core", "matmul_multi_core")
MATMUL_HEADER = "matmul_common/bmm_op.hpp"
GATE = "add_2_integers_in_riscv"
LABEL = "simulator wall time, exploratory, sizing only"
# The examples' own success lines (plans/spikes/p4-ttsim-runtime.md), and SYNTHETIC PCC lines in the printed form.
GATE_SUCCESS = "Success: Result is 21\n"
PASSED = "Test Passed\n"
SYNTHETIC_PCC = "0.98765"
PCC_LINE = f"Metalium vs Golden -- PCC = {SYNTHETIC_PCC}\n"
LOW_PCC_LINE = "Metalium vs Golden -- PCC = 0.96\n"


def passing_stdout(example: str) -> str:
    """Return SYNTHETIC stdout on which `example` passes its own check."""
    if example == GATE:
        return "SYNTHETIC log line\n" + GATE_SUCCESS
    if example in MATMULS:
        return "SYNTHETIC log line\n" + PCC_LINE + PASSED
    return "SYNTHETIC log line\n" + PASSED


# ---------------------------------------------------------------------------
# Fakes


def kernel_path(example: str) -> str:
    """Return the SYNTHETIC kernel path the stand-in host program names, relative to EXAMPLES_DIR."""
    return f"{EXAMPLES[example]}/kernels/synthetic_kernel.cpp"


def host_source(example: str) -> str:
    """Return the SYNTHETIC stand-in for an example's host program: the kernel-prefix fallback, a named kernel."""
    include = f'#include "{MATMUL_HEADER}"\n' if example in MATMULS else ""
    return (
        f"// SYNTHETIC stand-in for {example}'s host program, a test fixture; never compiled\n"
        "#ifndef OVERRIDE_KERNEL_PREFIX\n"
        '#define OVERRIDE_KERNEL_PREFIX ""\n'
        "#endif\n"
        f"{include}"
        f'static const char* kKernel = OVERRIDE_KERNEL_PREFIX "{kernel_path(example)}";\n'
        "int main() { return 0; }\n"
    )


def write_examples(tree: Path) -> None:
    """Write the SYNTHETIC stand-ins of the six examples, their kernels, and the matmul header into the tree."""
    examples = tree / EXAMPLES_DIR
    for example, directory in EXAMPLES.items():
        (examples / directory / "kernels").mkdir(parents=True, exist_ok=True)
        (examples / directory / f"{example}.cpp").write_bytes(host_source(example).encode("ascii"))
        (examples / kernel_path(example)).write_bytes(f"// SYNTHETIC kernel of {example}\n".encode("ascii"))
    header = examples / "matmul" / "matmul_common" / "bmm_op.hpp"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.write_bytes(b"// SYNTHETIC stand-in for Matmul::Common's header\n")


@dataclass
class Build:
    """One build the fake toolchain was asked for."""

    files: dict[str, str]
    workdir: Path
    harness: dict[str, str] | None


@dataclass
class Builds:
    """What the fake toolchain saw, and whether its builds fail."""

    fail: bool = False
    calls: list[Build] = field(default_factory=list)


class FakeHostToolchain:
    """Stands in for ttmetal-host: records each build and writes a PLACEHOLDER artifact, or fails with an error."""

    name = "ttmetal-host"
    capabilities = frozenset({"diagnostics", "emits_warnings"})

    def __init__(self, builds: Builds, tree: Path) -> None:
        """Keep the record and the tree, as TtMetalHost keeps its tree."""
        self.builds = builds
        self.tree = tree

    def build(self, files: Mapping[str, str], workdir: Path, harness: Mapping[str, str] | None = None) -> BuildResult:
        """Record the build and write the files; return the artifact, or one compile error when builds fail."""
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        self.builds.calls.append(Build(dict(files), workdir, None if harness is None else dict(harness)))
        for relative, text in [*files.items(), *(harness or {}).items()]:
            target = workdir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(text.encode("utf-8"))
        (workdir / "compile.stderr").write_bytes(b"")
        if self.builds.fail:
            error = Diagnostic(stage="compile", severity="error", message="SYNTHETIC compile error")
            return BuildResult(artifact=None, diagnostics=[error])
        artifact = workdir / "main"
        artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain, never executed\n")
        return BuildResult(artifact=artifact, diagnostics=[])


@dataclass
class Program:
    """What the fake sandbox answers for every program run, and the calls it saw."""

    stdout: str
    returncode: int = 0
    hang: bool = False
    stderr: str = ""
    calls: list[tuple[Any, list[str], Limits]] = field(default_factory=list)

    def run(self, spec: Any, argv: Sequence[str], limits: Limits) -> Any:
        """Record the call and return the canned SandboxResult; nothing is executed."""
        self.calls.append((spec, list(argv), limits))
        return sandbox_module.SandboxResult(
            returncode=self.returncode, stdout=self.stdout, stderr=self.stderr, wall_s=0.25, hang=self.hang,
            killed=False,
        )


@dataclass
class Host:
    """The faked host: the runs root, the install, the builds, and the program's answer."""

    runs_root: Path
    install: FakeInstall
    builds: Builds
    program: Program


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Host:
    """Set the gate's variables, the PLACEHOLDER install with the example stand-ins, and the fakes."""
    runs = tmp_path / "runs"
    runs.mkdir()
    (tmp_path / "scratch" / "tmp").mkdir(parents=True)
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scratch" / "tmp"))
    install = make_install(tmp_path, monkeypatch)
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(install.root))
    write_examples(install.tree)
    builds = Builds()

    def fake_build_toolchain(name: str, root: Path | None, *args: object, **kwargs: object) -> BuiltToolchain:
        """Return ttmetal-host as build_toolchain would, with the fake toolchain; nothing is compiled or run."""
        assert name == "ttmetal-host", name
        if root is None:
            raise RunError("SYNTHETIC: no toolchains root is set; set LASSI_TOOLCHAINS")
        assert same_path(root, install.root), "the toolchains root is $LASSI_TOOLCHAINS"
        environment = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
        pins = {"tt-metal": read_pin("tt-metal")}
        toolchain = FakeHostToolchain(builds, install.tree)
        return BuiltToolchain(name, (), toolchain, "/usr/bin/clang++-20", environment, pins, ("PLACEHOLDER",), 0)

    monkeypatch.setattr(runner_module, "build_toolchain", fake_build_toolchain)
    program = Program(stdout=passing_stdout(GATE))
    monkeypatch.setattr(sandbox_module.Sandbox, "run", lambda self, spec, argv, limits: program.run(spec, argv, limits))
    return Host(runs_root=runs, install=install, builds=builds, program=program)


def _drop_tool_modules() -> None:
    """Forget the tool and any module it loaded from tools/, so the next import sees the current fakes."""
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and Path(path).resolve().parent == TOOLS_DIR:
            del sys.modules[name]


@pytest.fixture
def tool(host: Host, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Import the smoke tool fresh from tools/, after the fakes are in place; forget it afterwards."""
    if str(TOOLS_DIR) not in sys.path:
        monkeypatch.syspath_prepend(str(TOOLS_DIR))
    _drop_tool_modules()
    module = importlib.import_module(TOOL_MODULE)
    if hasattr(module, "build_toolchain"):
        monkeypatch.setattr(module, "build_toolchain", runner_module.build_toolchain)
    yield module
    _drop_tool_modules()


@dataclass(frozen=True)
class ToolRun:
    """How one call of the tool's main() ended: its status and what it printed."""

    status: int
    out: str
    err: str


def run_tool(tool: ModuleType, argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> ToolRun:
    """Call the tool's main(argv) and return its status (a SystemExit counts) and its printed output."""
    try:
        status: Any = tool.main(list(argv))
    except SystemExit as exit_:
        status = exit_.code
    captured = capsys.readouterr()
    if isinstance(status, str):
        return ToolRun(1, captured.out, captured.err + status)
    return ToolRun(0 if status is None else int(status), captured.out, captured.err)


def summaries(runs_root: Path) -> list[Path]:
    """Return every summary.json the tool wrote under the runs root's ttsim-smoke directory."""
    return sorted((runs_root / "ttsim-smoke").glob("*/summary.json"))


def read_summary(runs_root: Path) -> tuple[Path, str, dict[str, Any]]:
    """Return the one summary.json, its text (plain ASCII), and its data."""
    (path,) = summaries(runs_root)
    raw = path.read_bytes()
    assert raw.isascii(), "summary.json is plain ASCII"
    data = json.loads(raw.decode("ascii"))
    assert isinstance(data, dict)
    return path, raw.decode("ascii"), data


def exit_status(summary: Mapping[str, Any]) -> Any:
    """Return the run's exit status as the summary records it."""
    assert "exit_code" in summary or "exit_status" in summary, sorted(summary)
    return summary.get("exit_code", summary.get("exit_status"))


def reads_as_pass(verdict: Any) -> bool:
    """Return True when a verdict reads as a pass (True, or a word starting with "pass")."""
    return verdict is True or (isinstance(verdict, str) and verdict.lower().startswith("pass"))


# ---------------------------------------------------------------------------
# Passing runs and the summary


def test_the_gate_example_passes_and_writes_its_summary(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    run = run_tool(tool, [GATE], capsys)
    assert run.status == 0, run.err
    path, raw, summary = read_summary(host.runs_root)
    assert reads_as_pass(summary["verdict"]), summary["verdict"]
    assert exit_status(summary) == 0
    assert (summary["hang"], summary["sim_ub"], summary["sim_gap"]) == (False, False, None)
    assert "Success: Result is 21" in raw, "the success line found"
    assert any(form in run.out for form in (str(path), path.as_posix())), "it prints the summary's path"


def test_the_summary_records_its_provenance(tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_tool(tool, [GATE], capsys).status == 0
    _, raw, summary = read_summary(host.runs_root)
    for key in ("commit", "dirty", "pins", "device", "host", "date", "argv", "findings", "verdict"):
        assert key in summary, key
    assert summary["commit"] is None or re.fullmatch(r"[0-9a-f]{40}", summary["commit"]), summary["commit"]
    assert summary["dirty"] in (True, False, None)
    ttsim_pin, metal_pin = read_pin("ttsim"), read_pin("tt-metal")
    ttsim_entry, metal_entry = json.dumps(summary["pins"]["ttsim"]), json.dumps(summary["pins"]["tt-metal"])
    assert ttsim_pin["VERSION"] in ttsim_entry and ttsim_pin["SHA256"] in ttsim_entry, ttsim_entry
    assert metal_pin["VERSION"] in metal_entry, metal_entry
    from lassi.executors.ttsim import TtsimExecutor

    assert summary["device"] == TtsimExecutor().device(), "the device is the executor's: a simulator"
    assert isinstance(summary["host"], str) and summary["host"]
    date = datetime.fromisoformat(summary["date"].replace("Z", "+00:00"))
    assert date.utcoffset() == timedelta(0), f"the date is UTC: {summary['date']}"
    assert GATE in summary["argv"]
    assert "wall_s" in raw and LABEL in raw, "the wall time carries its label (Agent Rule 2)"


def test_the_gate_names_the_example_by_its_build_target_and_it_passes_as_the_bare_name_does(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    """P4.G runs `ttsim_smoke.py metal_example_<name>` (plans/p4-ttsim.md), the pinned build's target name."""
    run = run_tool(tool, [f"metal_example_{GATE}"], capsys)
    assert run.status == 0, run.err
    _, raw, summary = read_summary(host.runs_root)
    assert summary["example"] == GATE, "the summary names the bare example"
    assert summary["argv"] == [f"metal_example_{GATE}"], "the argv is kept as given"
    assert "Success: Result is 21" in raw


@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_each_example_passes_on_its_own_check(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str], example: str
) -> None:
    host.program.stdout = passing_stdout(example)
    run = run_tool(tool, [example], capsys)
    assert run.status == 0, run.err
    _, raw, summary = read_summary(host.runs_root)
    assert reads_as_pass(summary["verdict"])
    if example in MATMULS:
        assert SYNTHETIC_PCC in raw, "the PCC the example printed is recorded"


def test_the_example_is_read_from_the_pinned_tree_and_built_under_the_runs_root(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    example = "matmul_single_core"
    host.program.stdout = passing_stdout(example)
    assert run_tool(tool, [example], capsys).status == 0
    (build,) = host.builds.calls
    examples = host.install.tree / EXAMPLES_DIR
    given = {**build.files, **(build.harness or {})}
    assert build.files[f"{example}.cpp"] == host_source(example), "the host program, from the tree"
    assert given[kernel_path(example)] == (examples / kernel_path(example)).read_text(encoding="ascii"), "its kernel"
    header = (examples / "matmul" / "matmul_common" / "bmm_op.hpp").read_text(encoding="ascii")
    assert given[MATMUL_HEADER] == header, "Matmul::Common's header at the path the program includes"
    smoke_root = host.runs_root / "ttsim-smoke"
    (summary_path,) = summaries(host.runs_root)
    assert summary_path.parent.parent == smoke_root
    own = Path(os.path.realpath(summary_path.parent))
    assert own == Path(os.path.realpath(build.workdir)) or own in Path(os.path.realpath(build.workdir)).parents, (
        "the build lies in the smoke run's own directory"
    )


def test_the_run_goes_through_the_executor_with_16_cpus_in_the_build_directory(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_tool(tool, [GATE], capsys).status == 0
    (build,) = host.builds.calls
    (spec, argv, limits), *_ = host.program.calls
    assert limits.cpus == 16
    assert same_path(spec.workdir, build.workdir) and same_path(argv[0], build.workdir / "main")
    environment = dict(spec.environment)
    assert "TT_METAL_SIMULATOR" in environment and "HOME" not in environment, "the ttsim executor's environment"


def test_each_smoke_run_gets_a_new_directory(tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_tool(tool, [GATE], capsys).status == 0
    assert run_tool(tool, [GATE], capsys).status == 0
    found = summaries(host.runs_root)
    assert len(found) == 2 and found[0].parent != found[1].parent


# ---------------------------------------------------------------------------
# Failing runs


# Each case misses one condition. The hang case keeps exit status 0 (SYNTHETIC: a real hang ends with 124, 137, or
# 143) so that it checks the hang flag on its own, apart from the exit status.
FAILS = {
    "no-success-line": (GATE, "SYNTHETIC log line\nError: Expected result of 21, got 20\n", 0, False),
    "another-result": (GATE, "Success: Result is 20\n", 0, False),
    "nonzero-exit": (GATE, GATE_SUCCESS, 1, False),
    "hang": (GATE, GATE_SUCCESS, 0, True),
    "undefined-behavior": (
        GATE, GATE_SUCCESS + "[4500] ERROR: NonContractualBehavior: rv32_mem_rd: SYNTHETIC unaligned read\n", 0, False,
    ),
    "gap": (GATE, GATE_SUCCESS + "[4489] ERROR: UnsupportedFunctionality: SYNTHETIC_fn: SYNTHETIC gap\n", 0, False),
    "test-failed": ("loopback", "Test Failed\n", 0, False),
    "matmul-low-pcc": ("matmul_single_core", LOW_PCC_LINE + PASSED, 0, False),
    "matmul-no-pcc": ("matmul_multi_core", PASSED, 0, False),
}


@pytest.mark.parametrize("case", sorted(FAILS))
def test_a_run_that_misses_any_condition_fails(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    example, stdout, returncode, hang = FAILS[case]
    host.program.stdout, host.program.returncode, host.program.hang = stdout, returncode, hang
    run = run_tool(tool, [example], capsys)
    assert run.status == 1, f"{case}: {run.out}{run.err}"
    _, _, summary = read_summary(host.runs_root)
    assert not reads_as_pass(summary["verdict"]), summary["verdict"]
    if case == "undefined-behavior":
        assert summary["sim_ub"] is True and "NonContractualBehavior" in json.dumps(summary["findings"])
    if case == "gap":
        assert summary["sim_gap"] == "UnsupportedFunctionality"


def test_a_kernel_jit_error_fails(tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.program.stdout = capture_text("jit-error", "stdout.txt") + GATE_SUCCESS
    host.program.stderr = capture_text("jit-error", "stderr.txt")
    host.program.returncode = 0
    assert run_tool(tool, [GATE], capsys).status == 1, "a jit error fails the smoke whatever else the run printed"


def test_a_failed_build_fails_without_a_run(tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.builds.fail = True
    assert run_tool(tool, [GATE], capsys).status == 1
    assert host.program.calls == [], "nothing runs"


# ---------------------------------------------------------------------------
# Usage errors and refusals


@pytest.mark.parametrize(
    "argv",
    [[], ["hello_world_compute_kernel"], ["../add_2_integers_in_riscv"], [GATE, "loopback"],
     ["metal_example_hello_world_compute_kernel"], ["metal_example_"], ["metal_example_-h"],
     [GATE, "metal_example_--"]],
    ids=["no-example", "not-one-of-the-six", "a-path", "two-examples", "a-prefixed-other-example",
         "the-prefix-alone", "a-prefixed-option", "a-prefixed-option-end"],
)
def test_a_usage_error_exits_2(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert run_tool(tool, argv, capsys).status == 2
    assert host.builds.calls == [] and host.program.calls == []


@pytest.mark.parametrize("name", ["LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS"])
def test_a_missing_environment_variable_exits_2(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv(name)
    assert run_tool(tool, [GATE], capsys).status == 2
    assert host.program.calls == []


def test_a_missing_ttsim_install_exits_2(tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str]) -> None:
    host.install.library.unlink()
    assert run_tool(tool, [GATE], capsys).status == 2
    assert host.program.calls == [], "nothing runs"


def test_a_missing_tt_metal_install_exits_2(
    tool: ModuleType, host: Host, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real build_toolchain refuses a tree that is not the pinned install (or, off the build host, the pinned
    # compiler it cannot find) with RunError before any process starts.
    def refusing_build_toolchain(name: str, root: Path | None, *args: object, **kwargs: object) -> BuiltToolchain:
        """Refuse as build_toolchain refuses a missing install."""
        raise RunError(f"SYNTHETIC: toolchain {name!r} builds against a tree that is not installed under {root}")

    monkeypatch.setattr(runner_module, "build_toolchain", refusing_build_toolchain)
    if hasattr(tool, "build_toolchain"):
        monkeypatch.setattr(tool, "build_toolchain", refusing_build_toolchain)
    assert run_tool(tool, [GATE], capsys).status == 2
    assert host.program.calls == []


# ---------------------------------------------------------------------------
# The tool itself


def test_the_tool_is_documented_typed_and_ascii(tool: ModuleType) -> None:
    raw = Path(tool.__file__).read_bytes()
    assert raw.isascii()
    tree = ast.parse(raw.decode("ascii"))
    assert ast.get_docstring(tree)
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assert all(ast.get_docstring(node) for node in functions), [n.name for n in functions if not ast.get_docstring(n)]
    main = next(node for node in functions if node.name == "main")
    assert main.returns is not None and all(arg.annotation is not None for arg in main.args.args)
