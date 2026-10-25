"""Tests for the energy profiler's rapl backend (task P9.1): a CPU package's energy counter from powercap sysfs.

Bible: Component Interfaces (Profiler; supports_power), Result Record
(Profiles), Project Recipes (Notes, Profilers), Design Principle 9, Agent
Rules 1 and 2; plans/p9-ee.md, P9.1.

The contract these tests fix (lassi.profilers.rapl):

- Backend rapl reads device kind cpu and takes interval_ms and package
  (an integer of at least 0, a bool is not one; required, no default; a
  refusal is a ValueError naming profiler.package, before any file is
  read).
- RaplSource(package, root) finds the one package zone of the MSR control
  type, <root>/sys/class/powercap/intel-rapl: among its direct
  subdirectories (the top-level zones, intel-rapl:<k> on a host), the one
  whose name file reads package-<package>. A zone's directory name is never
  parsed; subzones (one level deeper, such as core) and the zones of the
  intel-rapl-mmio control type are not package zones. It measures energy:
  read() is energy_uj in joules (microjoules / 1e6), read anew at each
  read, and range_j is max_energy_range_uj in joules, where the counter
  wraps.
- Refused at build, as TelemetryUnavailable naming the host path and never
  the root it was read under: no control type directory, no package zone of
  that number or more than one, a max_energy_range_uj that is missing or
  not a whole number above 0, and an energy_uj that cannot be read (mode
  0400 on recent kernels, root only) or does not parse.
- A read whose file is missing, unreadable, not a whole number, or past the
  range is a TelemetryError, so that window's energy is null.
- record() names the package domain, its zone and file as host paths, the
  range, and the name of every zone of the control type (subzones too).
- Nothing opens a file outside sys/class/powercap, nothing reads an mmio
  zone, and nothing starts a process.

Every host root is SYNTHETIC; no value in this module is a measurement.
"""

from __future__ import annotations

import errno
import os
import pathlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from profiler_fakes import NO_TICK_MS, PhaseClock, attribute, opened_under, profilers_module, write_files

from lassi.core.record import Profile

CPU = {"kind": "cpu"}
POWERCAP = "sys/class/powercap"
# SYNTHETIC zone directories write each ':' of the kernel's zone names (intel-rapl:0) as '-', since the test host's
# file system may refuse ':' in a name; the source never parses a zone's directory name.
PACKAGE_0 = "intel-rapl/intel-rapl-0"
PACKAGE_1 = "intel-rapl/intel-rapl-1"
CORE_0 = "intel-rapl/intel-rapl-0/intel-rapl-0-0"
MMIO_0 = "intel-rapl-mmio/intel-rapl-mmio-0"
RANGE = "262143000000\n"
# SYNTHETIC zones: directory under the powercap class -> file -> text. The mmio zone repeats package-0's name.
ZONES: dict[str, dict[str, str]] = {
    "intel-rapl": {"enabled": "1\n"},
    PACKAGE_0: {"name": "package-0\n", "energy_uj": "1000000000\n", "max_energy_range_uj": RANGE},
    CORE_0: {"name": "core\n", "energy_uj": "7000000\n", "max_energy_range_uj": RANGE},
    PACKAGE_1: {"name": "package-1\n", "energy_uj": "500000000\n", "max_energy_range_uj": RANGE},
    MMIO_0: {"name": "package-0\n", "energy_uj": "9000000\n", "max_energy_range_uj": RANGE},
}
# The host paths a refusal or a record names.
CONTROL = "/sys/class/powercap/intel-rapl"
ZONE_0 = f"/{POWERCAP}/{PACKAGE_0}"
ENERGY_0 = f"{ZONE_0}/energy_uj"
RANGE_0 = f"{ZONE_0}/max_energy_range_uj"


def rapl() -> ModuleType:
    """Return lassi.profilers.rapl, failing clearly while task P9.1 has not added it."""
    profilers_module()
    return profilers_module("rapl")


def unavailable() -> type[BaseException]:
    """Return lassi.profilers.power.TelemetryUnavailable."""
    return attribute(profilers_module("power"), "TelemetryUnavailable")


def telemetry_error() -> type[BaseException]:
    """Return lassi.profilers.power.TelemetryError."""
    return attribute(profilers_module("power"), "TelemetryError")


def powercap_root(root: Path, zones: Mapping[str, Mapping[str, str]] = ZONES) -> Path:
    """Write a SYNTHETIC powercap tree under `root` (zone directory -> file -> text) and return `root`."""
    files = {f"{POWERCAP}/{zone}/{name}": text for zone, entries in zones.items() for name, text in entries.items()}
    files["proc/cpuinfo"] = "model name\t: SYNTHETIC CPU model 9000\n"
    return write_files(root, files)


def changed(zone: str, **files: str | None) -> dict[str, dict[str, str]]:
    """Return ZONES with `zone`'s files replaced (None removes one)."""
    zones = {name: dict(entries) for name, entries in ZONES.items()}
    for name, text in files.items():
        if text is None:
            zones[zone].pop(name, None)
        else:
            zones[zone][name] = text
    return zones


def source(root: Path, package: int = 0) -> Any:
    """Build the real RaplSource of `package` under `root`."""
    return attribute(rapl(), "RaplSource")(package=package, root=root)


def profiler(root: Path, clock: Any = None, **changes: Any) -> Any:
    """Build an EnergyProfiler on backend rapl under `root`, package 0 at NO_TICK_MS unless `changes` say otherwise."""
    energy = profilers_module("energy")
    config: dict[str, Any] = {"device": dict(CPU), "backend": "rapl", "interval_ms": NO_TICK_MS, "package": 0}
    config.update(root=root, **changes)
    if clock is not None:
        config["clock"] = clock
    return attribute(energy, "EnergyProfiler")(**config)


def unavailable_message(root: Path, **changes: Any) -> str:
    """Build the rapl profiler under `root`, which must raise TelemetryUnavailable; return its message."""
    with pytest.raises(unavailable()) as info:
        profiler(root, **changes)
    message = str(info.value)
    assert str(root) not in message and root.as_posix() not in message, f"the message names the test root: {message}"
    return message


def unreadable(monkeypatch: pytest.MonkeyPatch, target: Path) -> None:
    """Make reading `target` fail as a mode 0400 root-owned file fails for another user (EACCES)."""
    real = pathlib.Path.read_bytes
    blocked = os.path.normcase(os.path.abspath(target))

    def read_bytes(self: Path) -> bytes:
        if os.path.normcase(os.path.abspath(self)) == blocked:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", read_bytes)


def energy_file(root: Path, zone: str = PACKAGE_0) -> Path:
    """Return the energy_uj file of `zone` under `root`."""
    return root / POWERCAP / zone / "energy_uj"


# ---------------------------------------------------------------------------
# The source: the zone, the unit, and the range


def test_rapl_reads_the_package_counter_in_joules(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host")
    found = source(root, 0)
    assert found.measures == "energy" and found.label is None
    assert found.read() == 1000.0
    assert found.range_j == 262143.0
    assert source(root, 1).read() == 500.0


def test_rapl_finds_the_package_by_its_name_not_its_directory(tmp_path: Path) -> None:
    zones = changed(PACKAGE_0, name="package-1\n")
    zones[PACKAGE_1]["name"] = "package-0\n"
    root = powercap_root(tmp_path / "host", zones)
    assert source(root, 0).read() == 500.0, "package 0 is the zone whose name is package-0"


def test_rapl_reads_the_counter_anew_at_each_read(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host")
    found = source(root)
    energy_file(root).write_bytes(b"1250000000\n")
    assert found.read() == 1250.0


@pytest.mark.parametrize(
    "text", ["abc\n", "\n", "-5\n", "1.5e9\n", "262143000001\n"], ids=["word", "empty", "negative", "float", "past"]
)
def test_rapl_bad_or_out_of_range_text_is_a_failed_read(tmp_path: Path, text: str) -> None:
    root = powercap_root(tmp_path / "host")
    found = source(root)
    energy_file(root).write_bytes(text.encode("ascii"))
    with pytest.raises(telemetry_error()):
        found.read()


def test_rapl_missing_or_unreadable_counter_is_a_failed_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = powercap_root(tmp_path / "host")
    found = source(root)
    unreadable(monkeypatch, energy_file(root))
    with pytest.raises(telemetry_error(), match="energy_uj"):
        found.read()
    monkeypatch.undo()
    energy_file(root).unlink()
    with pytest.raises(telemetry_error()):
        found.read()


def test_rapl_record_names_the_domain_its_paths_the_range_and_every_zone(tmp_path: Path) -> None:
    record = source(powercap_root(tmp_path / "host")).record()
    assert record["domain"] == "package-0"
    assert record["zone"] == ZONE_0 and record["energy_file"] == ENERGY_0
    assert record["max_energy_range_uj"] == 262143000000
    assert record["zones"] == {
        f"/{POWERCAP}/{PACKAGE_0}": "package-0",
        f"/{POWERCAP}/{CORE_0}": "core",
        f"/{POWERCAP}/{PACKAGE_1}": "package-1",
    }, "the control type's zones and subzones; the mmio control type is not read"


# ---------------------------------------------------------------------------
# Refusals at build: what this host cannot measure, naming the file


def test_an_unreadable_energy_counter_refuses_at_build_naming_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = powercap_root(tmp_path / "host")
    unreadable(monkeypatch, energy_file(root))
    message = unavailable_message(root)
    assert ENERGY_0 in message and "0400" in message, message


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs POSIX file modes and a user other than root")
def test_an_energy_counter_without_read_permission_refuses_at_build(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host")
    energy_file(root).chmod(0o000)
    try:
        assert ENERGY_0 in unavailable_message(root)
    finally:
        energy_file(root).chmod(0o600)


def test_no_control_type_directory_refuses_at_build(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host", {MMIO_0: ZONES[MMIO_0]})
    assert CONTROL in unavailable_message(root)


def test_a_package_the_host_lacks_refuses_and_lists_the_packages_found(tmp_path: Path) -> None:
    message = unavailable_message(powercap_root(tmp_path / "host"), package=2)
    assert "package-2" in message and "package-0" in message and "package-1" in message, message


def test_two_zones_of_one_package_refuse(tmp_path: Path) -> None:
    zones = changed(PACKAGE_1, name="package-0\n")
    message = unavailable_message(powercap_root(tmp_path / "host", zones))
    assert "intel-rapl-0" in message and "intel-rapl-1" in message, message


@pytest.mark.parametrize("text", [None, "0\n", "many\n"], ids=["missing", "zero", "word"])
def test_a_bad_counter_range_refuses_at_build_naming_the_file(tmp_path: Path, text: str | None) -> None:
    root = powercap_root(tmp_path / "host", changed(PACKAGE_0, max_energy_range_uj=text))
    assert RANGE_0 in unavailable_message(root)


def test_a_bad_first_counter_reading_refuses_at_build(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host", changed(PACKAGE_0, energy_uj="SYNTHETIC\n"))
    assert ENERGY_0 in unavailable_message(root)


@pytest.mark.parametrize(
    ("changes", "key"),
    [
        ({"package": None}, "profiler.package"),
        ({"package": -1}, "profiler.package"),
        ({"package": True}, "profiler.package"),
        ({"package": "0"}, "profiler.package"),
        ({"device": {"kind": "rocm", "indices": [0]}}, "profiler.device"),
        ({"interval_ms": 0}, "profiler.interval_ms"),
    ],
    ids=["no-package", "negative", "bool", "text", "rocm", "zero-interval"],
)
def test_a_refused_setting_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, Any], key: str
) -> None:
    root = powercap_root(tmp_path / "host")
    opened = opened_under(monkeypatch, root)
    with pytest.raises(ValueError, match=key):
        profiler(root, **changes)
    assert opened == [], "a refused setting reads nothing"


# ---------------------------------------------------------------------------
# The profiler on the real source


def test_rapl_profiles_a_window_in_joules_and_watts(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host")
    clock = PhaseClock(100.0)
    built = profiler(root, clock)
    built.start()
    energy_file(root).write_bytes(b"1150000000\n")
    clock.now = 102.0
    assert built.stop() == Profile(runtime_s=2.0, avg_power_w=75.0, energy_j=150.0)


def test_rapl_unwraps_a_counter_that_wrapped_in_the_window(tmp_path: Path) -> None:
    zone = {"energy_uj": "990000000\n", "max_energy_range_uj": "1000000000\n"}
    root = powercap_root(tmp_path / "host", changed(PACKAGE_0, **zone))
    clock = PhaseClock(0.0)
    built = profiler(root, clock)
    built.start()
    energy_file(root).write_bytes(b"15000000\n")
    clock.now = 5.0
    assert built.stop() == Profile(runtime_s=5.0, avg_power_w=5.0, energy_j=25.0)


def test_a_counter_lost_in_the_window_nulls_its_energy(tmp_path: Path) -> None:
    root = powercap_root(tmp_path / "host")
    clock = PhaseClock(0.0)
    built = profiler(root, clock)
    built.start()
    energy_file(root).unlink()
    clock.now = 2.0
    assert built.stop() == Profile(runtime_s=2.0), "the failed stop read is never bridged"


def test_rapl_telemetry_record_reaches_the_profiler(tmp_path: Path) -> None:
    record = profiler(powercap_root(tmp_path / "host")).telemetry()
    assert (record["profiler"], record["backend"], record["measures"], record["label"]) == (
        "energy", "rapl", "energy", None,
    )
    assert record["domain"] == "package-0" and record["energy_file"] == ENERGY_0


def test_rapl_reads_only_powercap_files_and_starts_no_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = powercap_root(tmp_path / "host")

    def no_process(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"a process was started: {args!r}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(os, "system", no_process)
    opened = opened_under(monkeypatch, root)
    clock = PhaseClock(1.0)
    built = profiler(root, clock)
    built.start()
    clock.now = 2.0
    built.stop()
    powercap = os.path.normcase(os.path.abspath(root / POWERCAP))
    outside = [path for path in opened if not path.startswith(powercap + os.sep)]
    assert opened and outside == [], f"files read outside the powercap tree: {outside}"
    assert not [path for path in opened if "mmio" in path], "the mmio control type is never read"
    counters = {path for path in opened if path.endswith("energy_uj")}
    assert counters == {os.path.normcase(os.path.abspath(energy_file(root)))}, counters
