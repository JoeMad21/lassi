"""Tests for the tt_silicon executor's device-timing reading and its sandbox names (task P17.42).

Bible: Execution Backends (the tt_silicon row), Sandbox, Result Record
(RunInfo.device_timing), Agent Rules 1, 2, 9, and 10; plans/spikes/tt-perf.md,
design (A); the owner's answers of 2026-10-26: device timing from the main
tt-metal pin built with Tracy (the spike's option (b) of OQ-105).

These tests fix:

- The executor takes a third setting, device_profiler, required with no
  default: "off" or "kernel" (lassi.executors.tt_profiler SETTINGS); any
  other value, or none, is a ValueError naming the key before any host
  file is read. "kernel" is refused at construction unless the pinned
  tt-metal is built with Tracy (toolchains/tt-metal.pin CMAKE_FLAGS holds
  -DENABLE_TRACY=ON), and before the program, with SandboxUnavailableError,
  unless the tree's install record names that flag on its flags line.
- With "kernel", the program's environment gains exactly the four
  variables of tt_profiler.profiler_environment, the profiler directory
  being <workdir>/@tt-silicon/profiler, emptied of an earlier run's files
  before the run; with "off" it gains none.
- The run's RunResult.device_timing is the reading tt_profiler parses
  after the run, its build the pin's PREFIX_NAME; a shortfall gives None
  and one run-stage note coded device-timing; "off" gives None and no note.
  The profiler's files are never output files.
- The sandbox's fixed allowlist holds the four names (TT_PROFILER_NAMES),
  which a SandboxSpec may set only with a TenstorrentExposure, so no ttsim
  or native run can switch the device profiler on (Agent Rule 2), and a
  compile environment never.

Every host tree and CSV is SYNTHETIC; nothing runs a program and no value
is a measurement.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from test_sandbox import sample_spec
from test_sandbox_tenstorrent import gpu_exposure, make_tt
from test_tt_profiler import write_logs
from test_tt_silicon import INPUTS, LIMITS, ExplodingSandbox, Roots, silicon
from tt_silicon_fakes import PRIVATE_PER_RUN, Access, Host, write_host
from ttsim_fakes import make_artifact, make_install

from lassi.core.record import DeviceTiming
from lassi.executors import sandbox as sandbox_module
from lassi.executors.sandbox import SandboxUnavailableError
from lassi.toolchains import pins as pins_module

PROFILER_NAMES = (
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_PROFILER_CPP_POST_PROCESS",
    "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY",
    "TT_METAL_PROFILER_DIR",
)
MISSING = object()


@pytest.fixture(scope="module")
def module() -> ModuleType:
    """Import lassi.executors.tt_silicon."""
    return importlib.import_module("lassi.executors.tt_silicon")


@pytest.fixture(autouse=True)
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Roots:
    """Set the runs, scratch, and home roots and TMPDIR under tmp_path, and unset $LASSI_TOOLCHAINS."""
    found = Roots(tmp_path / "runs", tmp_path / "scratch", tmp_path / "home", tmp_path / "tmpdir")
    for path in (found.runs, found.scratch, found.home, found.tmpdir):
        path.mkdir()
    for name, path in (("LASSI_RUNS_ROOT", found.runs), ("LASSI_SCRATCH", found.scratch), ("HOME", found.home),
                       ("TMPDIR", found.tmpdir)):
        monkeypatch.setenv(name, str(path))
    monkeypatch.delenv("LASSI_TOOLCHAINS", raising=False)
    return found


@pytest.fixture
def host(tmp_path: Path) -> Host:
    """Write the SYNTHETIC host tree of eight cards under tmp_path/host."""
    return write_host(tmp_path / "host")


@pytest.fixture
def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the PLACEHOLDER toolchains root whose tt-metal tree the install check accepts (ttsim_fakes)."""
    return make_install(tmp_path, monkeypatch)


def profiler() -> ModuleType:
    """Return lassi.executors.tt_profiler."""
    return importlib.import_module("lassi.executors.tt_profiler")


def tracy_record(fake: Any, flags: str | None = None) -> None:
    """Append the install script's flags line to the fake tree's record: the pin's CMAKE_FLAGS unless given."""
    flags = pins_module.read_pin("tt-metal")["CMAKE_FLAGS"] if flags is None else flags
    with fake.record.open("ab") as record:
        record.write(f"flags {flags}\n".encode("ascii"))


def writes_profile(spec: Any, argv: list[str]) -> None:
    """A program's effect under the profiler: the SYNTHETIC CSVs where TT_METAL_PROFILER_DIR points."""
    write_logs(Path(dict(spec.environment)["TT_METAL_PROFILER_DIR"]))


# ---------------------------------------------------------------------------
# The setting


@pytest.mark.parametrize(
    "value",
    [MISSING, None, "", "on", "Kernel", "noc", 1, True],
    ids=["absent", "none", "empty", "on", "upper-case", "noc", "int", "bool"],
)
def test_device_profiler_is_required_and_takes_off_or_kernel_before_any_host_file(
    module: ModuleType, tmp_path: Path, value: Any
) -> None:
    empty = tmp_path / "empty"
    access = Access(empty)
    settings: dict[str, Any] = {"hugepages": PRIVATE_PER_RUN, "reset_filter": True}
    if value is not MISSING:
        settings["device_profiler"] = value
    with pytest.raises(ValueError) as caught:
        module.TtSiliconExecutor(**settings, root=empty, access=access, uid=4100, sandbox=ExplodingSandbox())
    message = str(caught.value)
    assert "device_profiler" in message and "off" in message and "kernel" in message, message
    assert access.asked == [], "the setting is refused before any host file is read"


def test_device_profiler_is_a_config_key(module: ModuleType) -> None:
    from lassi.core.registry import DEFAULT_REGISTRY

    assert "device_profiler" in DEFAULT_REGISTRY.get("Executor", "tt_silicon").config_keys


def test_kernel_is_refused_unless_the_pinned_tt_metal_is_built_with_tracy(
    module: ModuleType, host: Host, install: Any
) -> None:
    pin_file = install.pins_dir / "tt-metal.pin"
    text = pin_file.read_bytes().decode("utf-8")
    assert "-DENABLE_TRACY=ON" in text, "the pin builds tt-metal with Tracy (task P17.42)"
    pin_file.write_bytes(text.replace("-DENABLE_TRACY=ON", "-DENABLE_TRACY=OFF").encode("utf-8"))
    with pytest.raises(ValueError, match="ENABLE_TRACY"):
        silicon(module, host, install, device_profiler="kernel")
    silicon(module, host, install, device_profiler="off")


def test_kernel_needs_an_install_record_that_names_the_tracy_flag(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    tracy_record(install, "-G Ninja -DENABLE_TRACY=OFF")
    executor, fake = silicon(module, host, install, device_profiler="kernel")
    with pytest.raises(SandboxUnavailableError, match="ENABLE_TRACY"):
        executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert fake.calls == [], "nothing ran"


# ---------------------------------------------------------------------------
# The environment and the reading


def test_kernel_adds_the_four_profiler_variables_and_off_adds_none(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    tracy_record(install)
    artifact = make_artifact(roots.runs)
    workdir = Path(os.path.realpath(artifact.parent))
    executor, fake = silicon(module, host, install, device_profiler="kernel", effects=[writes_profile])
    executor.run(artifact, INPUTS, LIMITS)
    environment = fake.calls[0].environment
    directory = Path(environment["TT_METAL_PROFILER_DIR"])
    assert {name: environment[name] for name in PROFILER_NAMES} == profiler().profiler_environment(directory)
    assert Path(os.path.realpath(directory)).relative_to(workdir).parts == ("@tt-silicon", "profiler")
    executor, fake = silicon(module, host, install, device_profiler="off")
    executor.run(make_artifact(roots.runs / "second"), INPUTS, LIMITS)
    assert not set(PROFILER_NAMES) & set(fake.calls[0].environment)


def test_a_profiled_run_carries_the_reading_and_no_profiler_file_is_an_output(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    tracy_record(install)
    executor, _ = silicon(module, host, install, device_profiler="kernel", effects=[writes_profile])
    ran = executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert isinstance(ran.device_timing, DeviceTiming)
    assert ran.device_timing.build == pins_module.read_pin("tt-metal")["PREFIX_NAME"]
    assert ran.device_timing.aiclk_mhz == 1000 and len(ran.device_timing.programs) == 2
    assert [entry for entry in ran.diagnostics if entry.code == "device-timing"] == []
    assert ran.output_files == {}, "the profiler's files lie under the reserved directory"


def test_a_profiled_run_without_the_files_gives_no_reading_and_a_note(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    tracy_record(install)
    executor, _ = silicon(module, host, install, device_profiler="kernel")
    ran = executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert ran.device_timing is None
    (note,) = [entry for entry in ran.diagnostics if entry.code == "device-timing"]
    assert (note.stage, note.severity) == ("run", "note") and "not recorded" in note.message


def test_an_unprofiled_run_has_no_reading_and_no_note(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    executor, _ = silicon(module, host, install, device_profiler="off", effects=[writes_profile_if_set])
    ran = executor.run(make_artifact(roots.runs), INPUTS, LIMITS)
    assert ran.device_timing is None
    assert [entry for entry in ran.diagnostics if entry.code == "device-timing"] == []


def writes_profile_if_set(spec: Any, argv: list[str]) -> None:
    """An effect that writes the CSVs only when the profiler directory is set (it never is with off)."""
    directory = dict(spec.environment).get("TT_METAL_PROFILER_DIR")
    if directory is not None:
        write_logs(Path(directory))


def test_an_earlier_runs_profiler_files_are_gone_before_the_next_run(
    module: ModuleType, host: Host, install: Any, roots: Roots
) -> None:
    tracy_record(install)
    artifact = make_artifact(roots.runs)
    stale = Path(os.path.realpath(artifact.parent)) / "@tt-silicon" / "profiler" / ".logs"
    stale.mkdir(parents=True)
    (stale / "cpp_device_perf_report.csv").write_bytes(b"SYNTHETIC stale report of an earlier run\n")
    seen: list[bool] = []

    def checks(spec: Any, argv: list[str]) -> None:
        seen.append((stale / "cpp_device_perf_report.csv").exists())

    executor, _ = silicon(module, host, install, device_profiler="kernel", effects=[checks])
    ran = executor.run(artifact, INPUTS, LIMITS)
    assert seen == [False], "the profiler would append to an earlier run's report"
    assert ran.device_timing is None


def test_the_device_line_names_the_tracy_build(module: ModuleType, host: Host, install: Any) -> None:
    executor, _ = silicon(module, host, install)
    assert executor.device().endswith("; tt-metal 5280a9cf-tracy"), executor.device()
    from lassi.executors import ttsim

    assert "with tt-metal 5280a9cf-tracy: a simulator" in ttsim.TtsimExecutor().device()


def test_the_recorded_version_names_the_build_variant() -> None:
    assert pins_module.recorded_version({"VERSION": "5280a9cf", "BUILD": "tracy"}) == "5280a9cf-tracy"
    assert pins_module.recorded_version({"VERSION": "v1.3.4"}) == "v1.3.4"
    assert pins_module.recorded_version(pins_module.read_pin("tt-metal")) == "5280a9cf-tracy"


def test_a_trial_records_the_tt_metal_pin_with_its_build() -> None:
    from lassi.core import runner

    pins = runner._pin_versions([{"tt-metal": pins_module.read_pin("tt-metal")}])
    assert pins.tt_metal == "5280a9cf-tracy"


# ---------------------------------------------------------------------------
# The sandbox's names


def test_the_allowlist_holds_the_profiler_names_as_their_own_group() -> None:
    assert tuple(sandbox_module.TT_PROFILER_NAMES) == PROFILER_NAMES
    assert set(PROFILER_NAMES) <= sandbox_module.ENVIRONMENT_NAMES
    assert not set(PROFILER_NAMES) & set(sandbox_module.TT_METAL_NAMES), "the ttsim executor never sets them"


@pytest.mark.parametrize("name", PROFILER_NAMES)
def test_a_spec_sets_a_profiler_name_only_with_a_tenstorrent_exposure(name: str) -> None:
    environment = {"PATH": sandbox_module.SANDBOX_PATH, name: "1"}
    with pytest.raises(ValueError, match=name):
        sample_spec(sandbox_module, environment=environment)
    with pytest.raises(ValueError, match=name):
        sample_spec(sandbox_module, environment=environment, devices=gpu_exposure(sandbox_module))
    spec = sample_spec(sandbox_module, environment=environment, devices=make_tt(sandbox_module))
    assert dict(spec.environment)[name] == "1"


def test_a_compile_environment_never_sets_a_profiler_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="TT_METAL_DEVICE_PROFILER"):
        sandbox_module.SandboxedCompileRunner(
            environment={"PATH": sandbox_module.SANDBOX_PATH, "TT_METAL_DEVICE_PROFILER": "1"},
            toolchains=tmp_path,
            hidden_roots=[tmp_path / "hidden"],
        )
