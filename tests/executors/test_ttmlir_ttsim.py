"""Tests for the tt-mlir flatbuffer executor on ttsim, registered as Executor "ttmlir-ttsim" (task P17.48).

Bible: Execution Backends (the ttsim row), Harness Contract, Component
Interfaces (Executor; device(); pins), Agent Rules 2, 3, 6, 9, and 10.
Plan: plans/p17-portable.md, P17.48 (and P17.58, which runs TT-NN
flatbuffers through it); plans/spikes/ttmlir-runtime.md, "The LASSI driver".

The contract these tests fix:

- lassi.executors.ttmlir_ttsim.TtmlirTtsimExecutor is registered as
  Executor "ttmlir-ttsim", re-exported by the package, with the ttsim
  executor's capabilities, config keys, and keyword settings.
- run(flatbuffer, inputs, limits) goes through the sandbox only, with argv
  [driver, "run", flatbuffer, recorded descriptor, *inputs], the ttsim row's
  environment with TT_METAL_RUNTIME_ROOT on the runtime's own tt-metal tree
  and HOME on <workdir>/@ttsim/home, a reserved directory made before the
  run whose files are never outputs (the owner's answer to OQ-115), and the
  toolchains root mounted read-only; a Watcher rerun of a hang runs the
  driver on the copy's flatbuffer, with HOME in the copy's reserved
  directory.
- Before anything runs or is created in the workdir, the runtime install
  (record, library, tree) and the driver install (record, program,
  descriptor and its sha256, no PLACEHOLDER in the pin) are checked, each
  refusal a SandboxUnavailableError naming the path or key.
- The driver's "refused:" and "error:" lines and a death by signal become
  run-stage errors after the ttsim executor's findings.
- write_system_desc runs [driver, "describe", <SYSTEM_DESC>] in a directory
  under the runs root, needing no recorded descriptor.
- device() and pins name ttsim, the runtime (declared as tt-mlir-runtime and
  recorded under the field tt_mlir), and the driver, from the pins alone.

Every test uses a FakeSandbox and a PLACEHOLDER install (tests/executors/
ttsim_fakes.py); nothing is executed, and no value here is a measurement,
except the one finding line quoted from the exploratory run of 2026-10-30
(plans/spikes/ttmlir-runtime.md), used as text only.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from ttsim_fakes import FakeSandbox, canned, make_artifact, make_install, writes_watcher_log

from lassi.core import registry
from lassi.core.capabilities import SIMULATOR
from lassi.core.interfaces import Limits
from lassi.executors.sandbox import SANDBOX_PATH, SandboxUnavailableError
from lassi.toolchains import pins as pins_module

REPO = Path(__file__).resolve().parents[2]
LIMITS = Limits(wall_s=60.0, memory_mb=8192, cpus=4)
INPUTS = ["@inputs/a.lassiio", "@inputs/b.lassiio", "c.lassiio"]
DESC_BYTES = b"PLACEHOLDER: stands in for the recorded ttsim system descriptor in tests\n"
# A finding line of the exploratory TT-NN run of 2026-10-30 (rx 20261029-211133-...-e509), quoted as text.
UB_LINE = "[10918] ERROR: UndefinedBehavior: tensix_execute_unpacr: unpack_to_dst=0 in_data_format=8\n"
WATCHER_LOG = (
    "Dump #1 at 1.0s\n  core(x= 1,y= 1) k_ids:  1|  0|  0|  0|  0\nk_id[  1]: k.cpp\nDump #1 completed at 1.1s\n"
)


@dataclass(frozen=True)
class Fake:
    """The PLACEHOLDER install: the toolchains root and the paths the executor checks."""

    root: Path
    pins_dir: Path
    library: Path
    runtime: Path
    tree: Path
    runtime_library: Path
    driver_prefix: Path
    driver: Path
    desc: Path


def sha256(data: bytes) -> str:
    """Return the hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def rewrite(path: Path, values: dict[str, str]) -> None:
    """Replace the KEY=value line of each key in `values` in a pin file."""
    text = path.read_bytes().decode("ascii")
    for key, value in values.items():
        text, count = re.subn(rf"(?m)^{key}=.*$", f"{key}={value}", text)
        assert count == 1, key
    path.write_bytes(text.encode("ascii"))


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


def make_fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver_values: dict[str, str] | None = None) -> Fake:
    """Build the ttsim fake install, then the runtime and driver installs beside it, from patched pin copies."""
    base = make_install(tmp_path, monkeypatch)
    rewrite(base.pins_dir / "tt-mlir-driver.pin", {"SYSTEM_DESC_SHA256": sha256(DESC_BYTES), **(driver_values or {})})
    runtime, driver = pins_module.read_pin("tt-mlir-runtime"), pins_module.read_pin("tt-mlir-driver")
    prefix = base.root / runtime["PREFIX_NAME"]
    tree = prefix / runtime["METAL_DIR"]
    tree.mkdir(parents=True)
    head = " ".join(runtime[key] for key in ("NAME", "COMMIT", "URL", "VERSION", "BUILD"))
    (prefix / "lassi-install.txt").write_bytes(f"{head}\nversion-tag v0.0\n".encode("ascii"))
    library = prefix / runtime["RUNTIME_LIBRARY"]
    library.parent.mkdir(parents=True)
    library.write_bytes(b"PLACEHOLDER runtime library\n")
    driver_prefix = base.root / driver["PREFIX_NAME"]
    program = driver_prefix / driver["BINARY"]
    program.parent.mkdir(parents=True)
    program.write_bytes(b"PLACEHOLDER driver, never executed\n")
    values = [driver[key] for key in ("NAME", "VERSION", "SOURCE_SHA256", "HEADER_SHA256")]
    (driver_prefix / "lassi-install.txt").write_bytes(f"{' '.join([*values, runtime['PREFIX_NAME']])}\n".encode())
    desc = driver_prefix / driver["SYSTEM_DESC"]
    desc.write_bytes(DESC_BYTES)
    return Fake(base.root, base.pins_dir, base.library, prefix, tree, library, driver_prefix, program, desc)


def executor(fake: Fake, results: list[Any], effects: list[Any] | None = None) -> Any:
    """Return the executor on the fake install with a FakeSandbox of `results`."""
    from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor

    return TtmlirTtsimExecutor(toolchains=str(fake.root), sandbox=FakeSandbox(results, effects or []))


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fake:
    """Return the PLACEHOLDER install."""
    return make_fake(tmp_path, monkeypatch)


# ---------------------------------------------------------------------------
# Registration, device, pins


def test_registered_as_executor_ttmlir_ttsim() -> None:
    from lassi.executors import ttmlir_ttsim

    entry = registry.DEFAULT_REGISTRY.get("Executor", "ttmlir-ttsim")
    assert entry.factory is ttmlir_ttsim.TtmlirTtsimExecutor
    assert entry.capabilities == frozenset({"runs_code", "sandboxed", SIMULATOR})
    assert entry.config_keys == frozenset({"arch", "dispatch", "chips", "harness"})


def test_the_package_re_exports_it_and_its_docstring_lists_it() -> None:
    import lassi.executors as package

    assert package.TtmlirTtsimExecutor is package.ttmlir_ttsim.TtmlirTtsimExecutor
    assert {"TtmlirTtsimExecutor", "ttmlir_ttsim"} <= set(package.__all__)
    assert '"ttmlir-ttsim"' in (package.__doc__ or "")


def test_factory_works_with_no_argument_and_no_install() -> None:
    built = registry.DEFAULT_REGISTRY.get("Executor", "ttmlir-ttsim").factory()
    assert built.name == "ttmlir-ttsim"


@pytest.mark.parametrize(("key", "value"), [("arch", "blackhole"), ("dispatch", "fast"), ("chips", 2)])
def test_the_ttsim_rows_one_values_are_kept(key: str, value: object) -> None:
    from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor

    with pytest.raises(ValueError, match=key):
        TtmlirTtsimExecutor(**{key: value})


def test_device_names_ttsim_the_runtime_and_the_driver_as_a_simulator() -> None:
    from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor

    built = TtmlirTtsimExecutor()
    text = built.device()
    runtime, driver = pins_module.read_pin("tt-mlir-runtime"), pins_module.read_pin("tt-mlir-driver")
    assert text.isascii() and text.isprintable() and text == text.strip() and "\n" not in text
    assert "ttsim v1.3.4" in text and "a simulator, not silicon" in text
    assert pins_module.recorded_version(runtime) in text and runtime["COMMIT"][:8] in text
    assert runtime["METAL_COMMIT"][:8] in text and driver["VERSION"] in text


def test_pins_declare_ttsim_and_the_runtime_under_its_own_pin_name() -> None:
    from lassi.core.runner import pin_field
    from lassi.executors.ttmlir_ttsim import TtmlirTtsimExecutor

    built = TtmlirTtsimExecutor()
    assert set(built.pins) == {"ttsim", "tt-mlir-runtime"}, "not the compiler pin's name, tt-mlir"
    assert built.pins["tt-mlir-runtime"] == pins_module.read_pin("tt-mlir-runtime")
    assert pins_module.recorded_version(built.pins["tt-mlir-runtime"]) == "0.9.0.dev20260221-runtime"
    fields = {pin_field(name, pin) for name, pin in built.pins.items()}
    assert fields == {"ttsim", "tt_mlir"}, "the runtime build is recorded under tt-mlir's field"


# ---------------------------------------------------------------------------
# A run


def test_a_run_goes_through_the_driver_in_the_sandbox(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "add.ttm")
    sandbox_runs = executor(fake, [canned(stdout="lassi-ttmlir-run: ran program add (TTM0)\n")])
    result = sandbox_runs.run(artifact, INPUTS, LIMITS)
    (call,) = sandbox_runs.sandbox.calls
    assert call.argv == [str(fake.driver), "run", str(artifact), str(fake.desc), *INPUTS]
    assert call.spec.toolchains == fake.root and call.spec.workdir == artifact.parent
    environment = call.environment
    assert environment["TT_METAL_RUNTIME_ROOT"] == str(fake.tree), "the runtime's own tt-metal tree"
    assert environment["TT_METAL_SIMULATOR"] == str(fake.library)
    assert environment["PATH"] == SANDBOX_PATH
    home = artifact.parent / "@ttsim" / "home"
    assert environment["HOME"] == str(home) and home.is_dir(), "HOME is the reserved directory, made before the run"
    assert result.exit_code == 0 and result.diagnostics == [] and result.sim_ub is False


def test_files_written_under_home_are_never_outputs(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "add.ttm")

    def writes(spec: Any, argv: list[str]) -> None:
        home = Path(spec.environment["HOME"])
        (home / ".cache").mkdir()
        (home / ".cache" / "ttnn.cfg").write_bytes(b"PLACEHOLDER cache")
        (spec.workdir / "c.lassiio").write_bytes(b"PLACEHOLDER output")

    result = executor(fake, [canned()], [writes]).run(artifact, INPUTS, LIMITS)
    assert sorted(result.output_files) == ["c.lassiio"]


def test_the_outputs_are_the_new_files_outside_the_reserved_directory(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "add.ttm")

    def writes(spec: Any, argv: list[str]) -> None:
        (spec.workdir / "c.lassiio").write_bytes(b"PLACEHOLDER output")

    built = executor(fake, [canned()], [writes])
    result = built.run(artifact, INPUTS, LIMITS)
    assert sorted(result.output_files) == ["c.lassiio"]


def test_a_simulator_finding_and_a_driver_error_are_both_read(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "histo.ttnn")
    stderr = "lassi-ttmlir-run: error: TT_THROW in the runtime\n"
    built = executor(fake, [canned(returncode=1, stdout=UB_LINE, stderr=stderr)])
    result = built.run(artifact, INPUTS, LIMITS)
    codes = [(item.stage, item.code) for item in result.diagnostics]
    assert codes == [("run", "UndefinedBehavior"), ("run", "runtime-error")]
    assert result.sim_ub is True
    assert result.diagnostics[1].message == "the tt-mlir runtime failed: TT_THROW in the runtime"


def test_a_refusal_line_is_a_run_error(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "add.ttm")
    stderr = "noise\nlassi-ttmlir-run: refused: the flatbuffer was compiled for another system descriptor\n"
    result = executor(fake, [canned(returncode=2, stderr=stderr)]).run(artifact, INPUTS, LIMITS)
    (item,) = result.diagnostics
    assert (item.stage, item.severity, item.code) == ("run", "error", "driver-refused")
    assert "another system descriptor" in item.message


def test_a_signal_is_a_run_error_naming_it(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "add.ttm")
    result = executor(fake, [canned(returncode=139)]).run(artifact, INPUTS, LIMITS)
    (item,) = result.diagnostics
    assert item.code == "signal" and "SIGSEGV" in item.message and "139" in item.message
    assert result.exit_code == 139, "the exit status stays the sandbox's"


def test_a_hang_reruns_the_driver_with_watcher_on_the_copy(fake: Fake, runs_root: Path) -> None:
    artifact = make_artifact(runs_root, "add.ttm")
    built = executor(fake, [canned(returncode=124, hang=True), canned()], [None, writes_watcher_log(WATCHER_LOG)])
    result = built.run(artifact, INPUTS, LIMITS)
    first, rerun = built.sandbox.calls
    copy = artifact.parent / "@ttsim" / "watcher"
    assert rerun.argv == [str(fake.driver), "run", str(copy / "add.ttm"), str(fake.desc), *INPUTS]
    assert rerun.environment["TT_METAL_WATCHER"] == "1"
    assert rerun.environment["HOME"] == str(copy / "@ttsim" / "home")
    assert [item.code for item in result.diagnostics] == ["watcher"], "no signal error for a hang"


# ---------------------------------------------------------------------------
# Install checks: each refusal comes before anything runs or is created


def refused(fake: Fake, runs_root: Path) -> str:
    """Run once and return the refusal's message; check that nothing ran and nothing was created."""
    artifact = make_artifact(runs_root, "add.ttm")
    built = executor(fake, [])
    with pytest.raises(SandboxUnavailableError) as caught:
        built.run(artifact, INPUTS, LIMITS)
    assert built.sandbox.calls == []
    assert not (artifact.parent / "@ttsim").exists()
    return str(caught.value)


def test_a_runtime_record_for_another_build_is_refused(fake: Fake, runs_root: Path) -> None:
    (fake.runtime / "lassi-install.txt").write_bytes(b"tt-mlir 5f396fd6 url 0.9.0.dev20260221\n")
    assert str(fake.runtime) in refused(fake, runs_root)


def test_a_missing_runtime_library_is_refused(fake: Fake, runs_root: Path) -> None:
    fake.runtime_library.unlink()
    assert str(fake.runtime_library) in refused(fake, runs_root)


def test_a_missing_runtime_tree_is_refused(fake: Fake, runs_root: Path) -> None:
    fake.tree.rmdir()
    assert str(fake.tree) in refused(fake, runs_root)


def test_a_driver_record_for_another_source_is_refused(fake: Fake, runs_root: Path) -> None:
    (fake.driver_prefix / "lassi-install.txt").write_bytes(b"lassi-ttmlir-driver 000000000000 x y z\n")
    assert str(fake.driver_prefix) in refused(fake, runs_root)


def test_a_missing_driver_program_is_refused(fake: Fake, runs_root: Path) -> None:
    fake.driver.unlink()
    assert str(fake.driver) in refused(fake, runs_root)


def test_a_descriptor_with_other_bytes_is_refused(fake: Fake, runs_root: Path) -> None:
    fake.desc.write_bytes(b"another descriptor\n")
    message = refused(fake, runs_root)
    assert str(fake.desc) in message and sha256(DESC_BYTES) in message


def test_no_recorded_descriptor_refuses_every_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = make_fake(tmp_path, monkeypatch, {"SYSTEM_DESC_SHA256": "PLACEHOLDER"})
    assert "SYSTEM_DESC_SHA256" in refused(fake, tmp_path / "runs")


def test_a_placeholder_in_the_driver_pin_refuses_every_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = make_fake(tmp_path, monkeypatch, {"VERSION": "PLACEHOLDER"})
    assert "PLACEHOLDER" in refused(fake, tmp_path / "runs")


# ---------------------------------------------------------------------------
# The descriptor


def test_write_system_desc_runs_describe_mode_without_a_recorded_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runs_root: Path
) -> None:
    fake = make_fake(tmp_path, monkeypatch, {"SYSTEM_DESC_SHA256": "PLACEHOLDER"})
    fake.desc.unlink()
    workdir = runs_root / "describe"
    workdir.mkdir()

    def writes(spec: Any, argv: list[str]) -> None:
        (spec.workdir / "ttsim.ttsys").write_bytes(b"PLACEHOLDER descriptor")
        (spec.workdir / "other.txt").write_bytes(b"not an output")

    built = executor(fake, [canned()], [writes])
    run, written = built.write_system_desc(workdir, LIMITS)
    (call,) = built.sandbox.calls
    assert call.argv == [str(fake.driver), "describe", "ttsim.ttsys"]
    assert call.environment["HOME"] == str(workdir.resolve() / "@ttsim" / "home")
    assert call.environment["TT_METAL_RUNTIME_ROOT"] == str(fake.tree)
    assert written == workdir.resolve() / "ttsim.ttsys" and sorted(run.output_files) == ["ttsim.ttsys"]
    assert built.recorded_system_desc() == fake.desc.resolve() or built.recorded_system_desc() == fake.desc


def test_write_system_desc_refuses_a_directory_outside_the_runs_root(fake: Fake, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    with pytest.raises(ValueError, match="runs root"):
        executor(fake, []).write_system_desc(outside, LIMITS)


# ---------------------------------------------------------------------------
# The readers


def test_read_driver_diagnostics_reads_only_the_drivers_own_lines() -> None:
    from lassi.executors.ttmlir_ttsim import read_driver_diagnostics

    stderr = "x lassi-ttmlir-run: error: not at the start\nlassi-ttmlir-run: refused: no\nlassi-ttmlir-run: other\n"
    found = read_driver_diagnostics(stderr)
    assert [(item.code, item.message) for item in found] == [("driver-refused", "the driver refused the run: no")]


@pytest.mark.parametrize(
    ("status", "hang", "expected"),
    [(134, False, "SIGABRT"), (137, False, "SIGKILL"), (128 + 33, False, "signal 33"), (137, True, None),
     (0, False, None), (1, False, None), (128, False, None), (128 + 65, False, None), (None, False, None)],
)
def test_signal_diagnostic(status: int | None, hang: bool, expected: str | None) -> None:
    from lassi.executors.ttmlir_ttsim import signal_diagnostic

    found = signal_diagnostic(status, hang)
    if expected is None:
        assert found is None
    else:
        assert found is not None and expected in found.message and found.code == "signal"


def test_the_descriptor_tool_keeps_an_existing_file(tmp_path: Path) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("ttmlir_system_desc", REPO / "tools" / "ttmlir_system_desc.py")
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    written, target = tmp_path / "new.ttsys", tmp_path / "prefix" / "ttsim.ttsys"
    written.write_bytes(b"new")
    target.parent.mkdir()

    class Stand:
        """Stands in for the executor: only recorded_system_desc is read."""

        def recorded_system_desc(self) -> Path:
            return target

    assert tool._install(Stand(), written) is True and target.read_bytes() == b"new"
    written.write_bytes(b"other")
    assert tool._install(Stand(), written) is False and target.read_bytes() == b"new", "never replaced"
