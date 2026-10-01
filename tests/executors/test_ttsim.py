"""Tests for the ttsim executor, registered as Executor "ttsim" (task P4.11): construction, environment, install.

Bible: Execution Backends (the ttsim row), ttsim Facts, Sandbox, Component
Interfaces (Executor; device()), Agent Rules 2, 3, 6, 7, 9, 10, and 12.
Plan: plans/p4-ttsim.md, P4.11. The findings, the kernel JIT diagnostics,
and the Watcher rerun are in test_ttsim_findings.py; the remote checks are in
test_ttsim_remote.py.

The contract these tests fix:

- lassi.executors.ttsim.TtsimExecutor is registered as Executor "ttsim"
  (importing lassi.executors registers it, and the package re-exports it),
  with capabilities runs_code, sandboxed, and simulator and the config keys
  arch, dispatch, chips, and harness. Every constructor parameter is
  keyword-only: arch ("wormhole_b0"), dispatch ("slow"), chips (1), harness,
  hidden_roots, toolchains, and sandbox (None), so factory() works with no
  argument and no install. Any other arch, any other dispatch, and a chip
  count other than the int 1 (a bool is not an int) are refused at
  construction with ValueError naming the key.
- run(artifact, inputs, limits) runs only through the sandbox, as
  NativeExecutor.run does (the artifact's resolved directory inside the runs
  root is the workdir; the hidden roots plus the runs root; the harness and
  the toolchains root, $LASSI_TOOLCHAINS unless configured, read-only), and
  the program's environment is exactly the ttsim row's: PATH=SANDBOX_PATH,
  TMPDIR=/tmp, LANG=C, LC_ALL=C, TT_METAL_SIMULATOR on the pinned library,
  TT_METAL_SLOW_DISPATCH_MODE=1, TT_METAL_DISABLE_SFPLOADMACRO=1,
  TT_METAL_RUNTIME_ROOT on the pinned tree, TT_METAL_CACHE and
  TT_METAL_LOGS_PATH in the workdir's reserved directory @ttsim (created
  before the run), TT_METAL_INSPECTOR_RPC=0, and
  TT_METAL_THREADCOUNT=<Limits.cpus>, with no HOME and nothing from the
  caller's environment.
- Before anything runs, and before anything is created in the workdir, a
  missing library or descriptor, one whose sha256 is not the pin's, and a
  tree whose lassi-install.txt does not name the pin's NAME and COMMIT are
  refused with one error class (SandboxUnavailableError or ValueError, the
  implementer's choice) whose message names the path.
- output_files are the regular files new after the run, as native gives
  them, but none under @ttsim.
- device() names ttsim and its pin (VERSION and LIBRARY) and the pinned
  tt-metal (VERSION) as a simulator, from the pins alone: one line of
  printable ASCII, no process started, no sandbox used.
- The executor declares `pins`, pin name -> the pin file's pairs, as
  BuiltToolchain.pins holds them, so the runner can record ToolchainPins.ttsim.

Every test uses a FakeSandbox (tests/executors/ttsim_fakes.py) and a
PLACEHOLDER install; nothing is executed, no simulator or device is opened,
and no value in this module is a measurement.
"""

from __future__ import annotations

import ast
import inspect
import os
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from ttsim_fakes import (
    RESERVED,
    FakeInstall,
    FakeSandbox,
    canned,
    make_artifact,
    make_install,
    names_path,
    same_path,
)

from lassi.core import registry
from lassi.core.capabilities import SIMULATOR
from lassi.core.interfaces import Limits, RunResult
from lassi.core.record import TOOLCHAIN_PIN_NAMES
from lassi.executors.sandbox import SANDBOX_PATH, SandboxUnavailableError
from lassi.toolchains.pins import read_pin

LIMITS = Limits(wall_s=30.0, memory_mb=4096, cpus=3)
INPUTS = ["--size", "8", "two words"]
FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)
PROJECT_NAMES = ("lassi-repro", "lassi-ee", "lassi-df", "hecbench", "qwen", "wizardcoder", "a100", "mi300x")
# The ttsim row's fixed values, and the names whose values are paths.
FIXED = {
    "PATH": SANDBOX_PATH,
    "TMPDIR": "/tmp",
    "LANG": "C",
    "LC_ALL": "C",
    "TT_METAL_SLOW_DISPATCH_MODE": "1",
    "TT_METAL_DISABLE_SFPLOADMACRO": "1",
    "TT_METAL_INSPECTOR_RPC": "0",
}
PATH_NAMES = ("TT_METAL_SIMULATOR", "TT_METAL_RUNTIME_ROOT", "TT_METAL_CACHE", "TT_METAL_LOGS_PATH")
REFUSAL = (SandboxUnavailableError, ValueError)


@pytest.fixture
def ttsim() -> ModuleType:
    """Return the lassi.executors.ttsim module (imported here so each test shows a missing module clearly)."""
    from lassi.executors import ttsim

    return ttsim


@pytest.fixture(autouse=True)
def runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make tmp_path/runs the runs root, set the scratch and home roots beside it, and unset $LASSI_TOOLCHAINS."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
    monkeypatch.setenv("LASSI_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return runs


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeInstall:
    """Return the PLACEHOLDER install the executor's checks accept (ttsim_fakes.make_install)."""
    return make_install(tmp_path, monkeypatch)


def listing(root: Path) -> list[str]:
    """Return every path under `root`, relative and sorted."""
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def public_defs(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    """Return public top-level classes and functions, plus the public methods of public classes."""
    found: list[tuple[str, ast.AST]] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, *FUNCTION_NODES)) or node.name.startswith("_"):
            continue
        found.append((node.name, node))
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, FUNCTION_NODES) and not item.name.startswith("_"):
                    found.append((f"{node.name}.{item.name}", item))
    return found


def is_typed(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Return True when every parameter except self or cls, and the return value, are annotated."""
    args = node.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    params += [a for a in (args.vararg, args.kwarg) if a is not None]
    params = [p for p in params if p.arg not in ("self", "cls")]
    return node.returns is not None and all(p.annotation is not None for p in params)


# ---------------------------------------------------------------------------
# Registration and construction


def test_registered_as_executor_ttsim(ttsim: ModuleType) -> None:
    entry = registry.DEFAULT_REGISTRY.get("Executor", "ttsim")
    assert entry.factory is ttsim.TtsimExecutor
    assert entry.capabilities == frozenset({"runs_code", "sandboxed", SIMULATOR})
    assert entry.config_keys == frozenset({"arch", "dispatch", "chips", "harness"}), "the lassi-df block's keys"
    assert ttsim.TtsimExecutor.name == "ttsim"


def test_package_imports_and_re_exports_ttsim(ttsim: ModuleType) -> None:
    import lassi.executors as package

    assert package.ttsim is ttsim, "importing lassi.executors alone registers the executor"
    assert package.TtsimExecutor is ttsim.TtsimExecutor
    assert "TtsimExecutor" in package.__all__


def test_the_constructor_takes_keyword_settings_with_the_ttsim_rows_defaults(ttsim: ModuleType) -> None:
    params = inspect.signature(ttsim.TtsimExecutor).parameters
    assert set(params) == {"arch", "dispatch", "chips", "harness", "hidden_roots", "toolchains", "sandbox"}
    assert all(param.kind is inspect.Parameter.KEYWORD_ONLY for param in params.values())
    assert (params["arch"].default, params["dispatch"].default, params["chips"].default) == ("wormhole_b0", "slow", 1)
    assert all(params[name].default is None for name in ("harness", "hidden_roots", "toolchains", "sandbox"))
    assert list(inspect.signature(ttsim.TtsimExecutor.run).parameters) == ["self", "artifact", "inputs", "limits"]


def test_factory_works_with_no_arguments_and_no_install(ttsim: ModuleType) -> None:
    # The runner builds every bound executor and asks its device before any run, on any host.
    executor = registry.DEFAULT_REGISTRY.get("Executor", "ttsim").factory()
    assert executor.name == "ttsim"
    ttsim.TtsimExecutor(arch="wormhole_b0", dispatch="slow", chips=1)


@pytest.mark.parametrize("arch", ["blackhole", "grayskull", ""])
def test_another_arch_is_refused_at_construction(ttsim: ModuleType, arch: str) -> None:
    with pytest.raises(ValueError) as caught:
        ttsim.TtsimExecutor(arch=arch)
    assert "arch" in str(caught.value) and "wormhole_b0" in str(caught.value), "the key and the one supported value"


@pytest.mark.parametrize("dispatch", ["fast", ""])
def test_another_dispatch_is_refused_at_construction(ttsim: ModuleType, dispatch: str) -> None:
    with pytest.raises(ValueError) as caught:
        ttsim.TtsimExecutor(dispatch=dispatch)
    assert "dispatch" in str(caught.value) and "slow" in str(caught.value), "the key and the one supported value"


@pytest.mark.parametrize("chips", [2, 0, -1, True, "1", 1.0], ids=["two", "zero", "negative", "bool", "str", "float"])
def test_a_chip_count_other_than_one_is_refused_at_construction(ttsim: ModuleType, chips: object) -> None:
    with pytest.raises(ValueError) as caught:
        ttsim.TtsimExecutor(chips=chips)
    assert "chips" in str(caught.value)


def test_hidden_roots_given_as_one_string_are_refused(ttsim: ModuleType) -> None:
    with pytest.raises(ValueError):
        ttsim.TtsimExecutor(hidden_roots="/scratch")


def test_module_is_documented_typed_ascii_and_names_no_project(ttsim: ModuleType) -> None:
    raw = Path(ttsim.__file__).read_bytes()
    assert raw.isascii()
    source = raw.decode("ascii")
    assert not [name for name in PROJECT_NAMES if name in source.lower()], "Agent Rule 3: no project in lassi/"
    tree = ast.parse(source)
    assert ast.get_docstring(tree)
    missing_doc = [name for name, node in public_defs(tree) if not ast.get_docstring(node)]
    assert not missing_doc, missing_doc
    untyped = [name for name, node in public_defs(tree) if isinstance(node, FUNCTION_NODES) and not is_typed(node)]
    assert not untyped, untyped


# ---------------------------------------------------------------------------
# What run hands to the sandbox


def test_run_hands_the_sandbox_the_workdir_argv_limits_and_read_only_mounts(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, tmp_path: Path
) -> None:
    artifact = make_artifact(runs_root)
    harness = tmp_path / "assets" / "harness"
    fake = FakeSandbox(results=[canned()])
    ttsim.TtsimExecutor(harness=str(harness), toolchains=str(install.root), sandbox=fake).run(artifact, INPUTS, LIMITS)
    (call,) = fake.calls
    assert same_path(call.spec.workdir, artifact.parent), "the workdir is the artifact's directory"
    assert call.argv[1:] == INPUTS and same_path(call.argv[0], artifact)
    assert call.limits == LIMITS
    assert call.spec.harness is not None and same_path(call.spec.harness, harness)
    assert call.spec.toolchains is not None and same_path(call.spec.toolchains, install.root)
    roots = [os.path.realpath(root) for root in call.spec.hidden_roots]
    expected = [os.path.realpath(tmp_path / name) for name in ("scratch", "home", "runs")]
    assert roots == expected, "the scratch and home roots, then the runs root"


def test_the_program_environment_is_exactly_the_ttsim_rows(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    fake = FakeSandbox(results=[canned()])
    ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(artifact, INPUTS, LIMITS)
    environment = fake.calls[0].environment
    assert set(environment) == {*FIXED, *PATH_NAMES, "TT_METAL_THREADCOUNT"}, "no HOME, no TT_METAL_WATCHER"
    assert {name: environment[name] for name in FIXED} == FIXED
    assert environment["TT_METAL_THREADCOUNT"] == str(LIMITS.cpus), "the thread pool follows Limits.cpus"
    workdir = artifact.parent
    paths = {
        "TT_METAL_SIMULATOR": install.library,
        "TT_METAL_RUNTIME_ROOT": install.tree,
        "TT_METAL_CACHE": workdir / RESERVED / "cache",
        "TT_METAL_LOGS_PATH": workdir / RESERVED / "logs",
    }
    for name, path in paths.items():
        assert Path(environment[name]).is_absolute(), name
        assert same_path(environment[name], path), f"{name}={environment[name]!r}, expected {path}"


def test_the_callers_environment_never_reaches_the_program(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    leaked = {
        "TT_METAL_WATCHER": "1",
        "TT_METAL_KERNEL_PATH": "/leaked/kernels",
        "TT_METAL_CACHE": "/leaked/cache",
        "TT_METAL_THREADCOUNT": "999",
        "LD_PRELOAD": "/leaked/lib.so",
        "LASSI_TEST_API_KEY": "SYNTHETIC-secret",
    }
    for name, value in leaked.items():
        monkeypatch.setenv(name, value)
    artifact = make_artifact(runs_root)
    fake = FakeSandbox(results=[canned()])
    ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(artifact, [], LIMITS)
    environment = fake.calls[0].environment
    assert not set(environment) & {"TT_METAL_WATCHER", "TT_METAL_KERNEL_PATH", "LD_PRELOAD", "LASSI_TEST_API_KEY"}
    assert environment["TT_METAL_THREADCOUNT"] == str(LIMITS.cpus)
    assert same_path(environment["TT_METAL_CACHE"], artifact.parent / RESERVED / "cache")
    assert "SYNTHETIC-secret" not in repr(environment)


def test_the_jit_cache_and_logs_directories_exist_when_the_program_starts(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    seen: dict[str, bool] = {}

    def program_starts(spec: Any, argv: list[str]) -> None:
        """Note whether the directories the environment names exist when the program would start."""
        environment = dict(spec.environment)
        seen.update({name: Path(environment[name]).is_dir() for name in ("TT_METAL_CACHE", "TT_METAL_LOGS_PATH")})

    fake = FakeSandbox(results=[canned()], effects=[program_starts])
    ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(artifact, [], LIMITS)
    assert seen == {"TT_METAL_CACHE": True, "TT_METAL_LOGS_PATH": True}


def test_the_toolchains_root_defaults_to_the_environment(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LASSI_TOOLCHAINS", str(install.root))
    fake = FakeSandbox(results=[canned()])
    ttsim.TtsimExecutor(sandbox=fake).run(make_artifact(runs_root), [], LIMITS)
    assert fake.calls[0].spec.toolchains is not None and same_path(fake.calls[0].spec.toolchains, install.root)
    assert same_path(fake.calls[0].environment["TT_METAL_SIMULATOR"], install.library)


def test_no_toolchains_root_is_refused_before_anything_runs(ttsim: ModuleType, runs_root: Path) -> None:
    artifact = make_artifact(runs_root)
    before = listing(artifact.parent)
    fake = FakeSandbox(results=[canned()])
    with pytest.raises(REFUSAL):
        ttsim.TtsimExecutor(sandbox=fake).run(artifact, [], LIMITS)
    assert fake.calls == [] and listing(artifact.parent) == before


def test_a_relative_artifact_or_one_outside_the_runs_root_is_refused_before_anything_runs(
    ttsim: ModuleType, install: FakeInstall, tmp_path: Path
) -> None:
    fake = FakeSandbox(results=[canned()])
    executor = ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake)
    with pytest.raises(ValueError):
        executor.run(Path("build") / "main", [], LIMITS)
    outside = make_artifact(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="runs root"):
        executor.run(outside, [], LIMITS)
    assert fake.calls == []


# ---------------------------------------------------------------------------
# The install checks


def break_install(install: FakeInstall, how: str) -> Path:
    """Break the PLACEHOLDER install as `how` names and return the path the refusal must name."""
    if how == "missing-library":
        install.library.unlink()
        return install.library
    if how == "missing-descriptor":
        install.descriptor.unlink()
        return install.descriptor
    if how == "library-not-the-pinned-bytes":
        install.library.write_bytes(b"PLACEHOLDER: other bytes than the pin's sha256\n")
        return install.library
    if how == "descriptor-not-the-pinned-bytes":
        install.descriptor.write_bytes(b"# PLACEHOLDER: other bytes than the pin's sha256\n")
        return install.descriptor
    if how == "tree-without-install-record":
        install.record.unlink()
        return install.tree
    if how == "record-names-another-commit":
        metal = read_pin("tt-metal")
        install.record.write_bytes(f"{metal['NAME']} {'0' * 40} {metal['URL']}\n".encode("ascii"))
        return install.tree
    if how == "cpm-sources-differ":
        listed = install.tree / "lassi-cpm-sources.txt"
        listed.write_bytes(listed.read_bytes() + b"PLACEHOLDER-package https://example.invalid/x.git 0\n")
        return listed
    raise AssertionError(how)


BREAKAGES = (
    "missing-library",
    "missing-descriptor",
    "library-not-the-pinned-bytes",
    "descriptor-not-the-pinned-bytes",
    "tree-without-install-record",
    "record-names-another-commit",
    "cpm-sources-differ",
)


@pytest.mark.parametrize("how", BREAKAGES)
def test_a_broken_install_is_refused_before_anything_runs_or_is_created(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path, how: str
) -> None:
    named = break_install(install, how)
    artifact = make_artifact(runs_root)
    before = listing(artifact.parent)
    fake = FakeSandbox(results=[canned()])
    with pytest.raises(REFUSAL) as caught:
        ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(artifact, [], LIMITS)
    assert names_path(str(caught.value), named), f"the refusal names {named}: {caught.value}"
    assert fake.calls == [], "nothing runs"
    assert listing(artifact.parent) == before, "nothing is created in the workdir before the checks pass"


def test_every_install_refusal_raises_the_same_error_class(
    ttsim: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    classes: set[type] = set()
    for number, how in enumerate(BREAKAGES):
        base = tmp_path / f"case{number}"
        base.mkdir()
        install = make_install(base, monkeypatch)
        break_install(install, how)
        runs = base / "runs"
        monkeypatch.setenv("LASSI_RUNS_ROOT", str(runs))
        artifact = make_artifact(runs)
        with pytest.raises(REFUSAL) as caught:
            ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=FakeSandbox(results=[])).run(artifact, [], LIMITS)
        classes.add(type(caught.value))
    assert len(classes) == 1, f"one error class for every install refusal, got {sorted(c.__name__ for c in classes)}"


# ---------------------------------------------------------------------------
# What run returns


def test_output_files_are_the_new_files_outside_the_reserved_directory(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    artifact = make_artifact(runs_root)
    workdir = artifact.parent

    def program_writes(spec: object, argv: list[str]) -> None:
        """Write outputs, a JIT cache file, and a log file, the way a tt-metal program run does."""
        (workdir / "out.bin").write_bytes(b"\x02")
        (workdir / "sub").mkdir()
        (workdir / "sub" / "new.txt").write_bytes(b"new\n")
        (workdir / RESERVED / "cache" / "kernel.elf").write_bytes(b"PLACEHOLDER JIT output\n")
        inspector = workdir / RESERVED / "logs" / "generated" / "inspector"
        inspector.mkdir(parents=True)
        (inspector / "kernels.yaml").write_bytes(b"# PLACEHOLDER\n")

    fake = FakeSandbox(results=[canned()], effects=[program_writes])
    result = ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(artifact, [], LIMITS)
    assert sorted(result.output_files) == ["out.bin", "sub/new.txt"]
    assert all(same_path(path, workdir / relative) for relative, path in result.output_files.items())


def test_a_clean_run_passes_the_sandbox_fields_through_and_reads_no_finding(
    ttsim: ModuleType, install: FakeInstall, runs_root: Path
) -> None:
    outcome = canned(
        returncode=0, stdout="SYNTHETIC program stdout\n", stderr="SYNTHETIC program stderr\n", wall_s=1.25,
        stdout_truncated=True, workdir_incomplete=True,
    )
    fake = FakeSandbox(results=[outcome])
    result = ttsim.TtsimExecutor(toolchains=str(install.root), sandbox=fake).run(make_artifact(runs_root), [], LIMITS)
    assert isinstance(result, RunResult)
    assert (result.exit_code, result.hang, result.stdout, result.stderr, result.wall_s) == (
        0, False, "SYNTHETIC program stdout\n", "SYNTHETIC program stderr\n", 1.25,
    )
    assert (result.stdout_truncated, result.stderr_truncated, result.workdir_incomplete) == (True, False, True)
    assert result.sim_ub is False, "the executor always checks, so no finding reads as False, not None"
    assert result.sim_gap is None and list(result.diagnostics) == []


# ---------------------------------------------------------------------------
# The device and the pins


def assert_one_printable_line(device: object) -> str:
    """Assert that `device` is one non-empty line of printable ASCII with no leading or trailing blank."""
    assert isinstance(device, str) and device, device
    assert device.isascii() and device.isprintable() and device.strip() == device, repr(device)
    return device


def test_device_names_ttsim_its_pin_and_the_pinned_tt_metal_as_a_simulator(ttsim: ModuleType) -> None:
    device = assert_one_printable_line(ttsim.TtsimExecutor().device())
    ttsim_pin, metal_pin = read_pin("ttsim"), read_pin("tt-metal")
    for value in ("ttsim", ttsim_pin["VERSION"], ttsim_pin["LIBRARY"], metal_pin["VERSION"]):
        assert value in device, f"{value!r} is not in the device {device!r}"
    assert "simulator" in device.lower(), "Agent Rule 2: a simulator result names the simulator as the device"


def test_device_comes_from_the_pins(ttsim: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    make_install(tmp_path, monkeypatch, ttsim_values={"VERSION": "v0.0.0-PLACEHOLDER"})
    device = assert_one_printable_line(ttsim.TtsimExecutor().device())
    assert "v0.0.0-PLACEHOLDER" in device and "v1.3.4" not in device, device


def test_device_starts_no_process_and_uses_no_sandbox(ttsim: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def trap(*args: object, **kwargs: object) -> None:
        """Fail the test: device() starts no process."""
        raise AssertionError("device() started a process")

    monkeypatch.setattr(subprocess, "Popen", trap)
    fake = FakeSandbox(results=[])
    assert_one_printable_line(ttsim.TtsimExecutor(sandbox=fake).device())
    assert fake.calls == []


def test_the_executor_declares_its_pins_as_toolchains_declare_theirs(ttsim: ModuleType) -> None:
    pins = ttsim.TtsimExecutor().pins
    assert pins["ttsim"] == read_pin("ttsim"), "pin name -> the pin file's pairs, as BuiltToolchain.pins holds them"
    if "tt-metal" in pins:
        assert pins["tt-metal"] == read_pin("tt-metal")
    fields = [name.replace("-", "_") for name in pins]
    assert all(field in TOOLCHAIN_PIN_NAMES for field in fields), f"each pin has a Trial toolchain_pins field: {fields}"
