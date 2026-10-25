"""The energy profiler's rapl backend (task P9.1): one CPU package's energy counter from the powercap sysfs class.

Its recipe section names device kind cpu, the package, and the sampling
interval: `profiler: {kind: energy, backend: rapl, device: {kind: cpu},
package: 0, interval_ms: 10}`. package is required, an integer of at least
0, with no default; the kind cpu device section takes no indices, so the
package is the profiler's own key.

Its source, RaplSource, starts no tool and opens no device node: it reads
sysfs text and directory listings only, under the host root. The MSR
control type is <root>/sys/class/powercap/intel-rapl; its direct
subdirectories are the top-level zones (intel-rapl:<k> on a host), each
with a name file, and the package zone is the one whose name reads
package-<package>. A zone's directory name is never parsed. Subzones (such
as core) lie one level deeper, and the intel-rapl-mmio control type, which
repeats a package's name on some hosts, is not read; subzones' names are
recorded, and no subzone is a package zone. The zone's energy_uj is a
cumulative counter in microjoules, read anew at each read, that wraps to 0
past max_energy_range_uj, also in microjoules, which is read once at build.

Known cases, not complete: on recent kernels energy_uj is mode 0400, so
only root reads it, and a build as another user is refused naming the file;
the whole package is measured, so other users' load on a shared host is in
every value; the native executor does not pin a run to the named package,
so on a host with more than one package (alpha01 has two) an attempt may
run on another, and the values are package-<package>'s, not the attempt's
(pinning or recording the run's cores is P9.6's); a wrapped delta is taken
as range - before + after, short by one counter unit per wrap if the raw
counter wraps at the range plus one unit. Unverified until P9.2 reads the host's tree and the Linux
source at its release: the layout above, the name format, that AMD
packages appear under the same control type, and how often the counter
updates.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from lassi.core.devices import DeviceSpec
from lassi.profilers.energy import MEASURES_ENERGY, Backend, TelemetrySource, register_backend
from lassi.profilers.power import TelemetryError, TelemetryUnavailable

# The MSR control type directory under the host root; its direct subdirectories are the top-level zones.
CONTROL = PurePosixPath("/sys/class/powercap/intel-rapl")
# The recipe path a refusal of the package names.
PACKAGE_PATH = "profiler.package"
_MICROJOULES_PER_JOULE = 1_000_000
# Why energy_uj may be unreadable, named in the refusal.
_MODE_HINT = "on recent kernels energy_uj is mode 0400, readable by root only"


def _under(root: Path, host: PurePosixPath) -> Path:
    """Return the host path `host` (absolute, POSIX) under the root `root`."""
    return root.joinpath(*host.parts[1:])


def checked_package(value: object) -> int:
    """Return the package number: an integer of at least 0 (a bool is not one); ValueError naming the key."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        given = "no value" if value is None else repr(value)
        raise ValueError(f"{PACKAGE_PATH} must be an integer package number of at least 0, not {given}; no default")
    return value


def _text(root: Path, host: PurePosixPath) -> str | None:
    """Return the stripped ASCII text of the sysfs file `host` under `root`, or None when it cannot be read."""
    try:
        return _under(root, host).read_bytes().decode("ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _subzones(root: Path, host: PurePosixPath) -> list[PurePosixPath]:
    """Return the zone directories directly under `host` (real directories holding a name file), sorted."""
    try:
        with os.scandir(_under(root, host)) as found:
            names = sorted(entry.name for entry in found if entry.is_dir(follow_symlinks=False))
    except OSError:
        return []
    return [host / name for name in names if _under(root, host / name / "name").is_file()]


def zone_names(root: Path) -> dict[str, str]:
    """Return every zone of the control type and its subzones as host path -> name ("" when unreadable)."""
    found: dict[str, str] = {}
    for zone in _subzones(root, CONTROL):
        found[str(zone)] = _text(root, zone / "name") or ""
        for subzone in _subzones(root, zone):
            found[str(subzone)] = _text(root, subzone / "name") or ""
    return found


def _package_zone(package: int, root: Path) -> PurePosixPath:
    """Return the host path of the one top-level zone named package-<package>; TelemetryUnavailable otherwise."""
    if not _under(root, CONTROL).is_dir():
        raise TelemetryUnavailable(f"{CONTROL} does not exist on this host, so no RAPL package energy can be read")
    zones = {zone: _text(root, zone / "name") for zone in _subzones(root, CONTROL)}
    wanted = f"package-{package}"
    matches = [zone for zone, name in zones.items() if name == wanted]
    if len(matches) != 1:
        packages = sorted(name for name in zones.values() if name and name.startswith("package-"))
        listed = ", ".join(str(zone) for zone in matches) if matches else ", ".join(packages) or "none"
        raise TelemetryUnavailable(f"{CONTROL} must hold exactly one zone named {wanted}; found {listed}")
    return matches[0]


def _microjoules(text: str | None) -> int | None:
    """Return a whole number of microjoules, or None for text that is not one."""
    return int(text) if text is not None and text.isdigit() else None


class RaplSource:
    """The energy counter of one CPU package from its powercap zone, read anew at each read (a TelemetrySource)."""

    measures: str = MEASURES_ENERGY
    label: str | None = None

    def __init__(self, *, package: int, root: Path) -> None:
        """Find package `package`'s zone under `root`, read its range and counter; TelemetryUnavailable naming why."""
        self._root = Path(root)
        self._zone = _package_zone(package, self._root)
        self._domain = f"package-{package}"
        self._energy = self._zone / "energy_uj"
        limit = self._zone / "max_energy_range_uj"
        range_uj = _microjoules(_text(self._root, limit))
        if range_uj is None or range_uj <= 0:
            raise TelemetryUnavailable(f"{limit} must hold a whole number of microjoules above 0")
        self._range_uj = range_uj
        self.range_j: float | None = range_uj / _MICROJOULES_PER_JOULE
        try:
            self.read()
        except TelemetryError as error:
            raise TelemetryUnavailable(f"{error}; this user cannot measure package energy here") from None

    def read(self) -> float:
        """Return the counter in joules (microjoules / 1e6); TelemetryError for a missing, unreadable, or bad value."""
        try:
            text = _under(self._root, self._energy).read_bytes().decode("ascii").strip()
        except PermissionError:
            raise TelemetryError(f"{self._energy} could not be read: PermissionError ({_MODE_HINT})") from None
        except (OSError, UnicodeDecodeError) as error:
            raise TelemetryError(f"{self._energy} could not be read: {type(error).__name__}") from None
        value = _microjoules(text)
        if value is None or value > self._range_uj:
            raise TelemetryError(f"{self._energy} holds {text[:40]!r}, not microjoules within the counter's range")
        return value / _MICROJOULES_PER_JOULE

    def record(self) -> dict[str, object]:
        """Return the domain, its zone and counter file as host paths, the range, and every zone's name."""
        return {
            "domain": self._domain,
            "zone": str(self._zone),
            "energy_file": str(self._energy),
            "max_energy_range_uj": self._range_uj,
            "zones": zone_names(self._root),
        }


def _open_rapl(spec: DeviceSpec, settings: Mapping[str, object], root: Path) -> TelemetrySource:
    """Check profiler.package, then open package's RaplSource under `root`."""
    package = checked_package(settings.get("package"))
    return RaplSource(package=package, root=root)


register_backend(
    Backend(name="rapl", kinds=frozenset({"cpu"}), keys=frozenset({"interval_ms", "package"}), open=_open_rapl)
)
