"""The energy profiler, registered as Profiler "energy" (task P9.1; OQ-088): one interface, a backend per device.

Its recipe section names a backend, with no default, and the backend's own
keys: `profiler: {kind: energy, backend: rapl, device: {kind: cpu},
package: 0, interval_ms: 10}` or `profiler: {kind: energy, backend: wall,
device: {kind: cpu}}`. It declares supports_power and takes_device, like the
power profilers (lassi.profilers.power), so the runner probes its device and
refuses a profiled executor that does not run there.

The seam (TelemetrySource) takes either a cumulative energy counter in
joules or a power reading in watts, or measures runtime only, and every
backend says which on this host when it is opened; a backend that cannot
measure refuses then (TelemetryUnavailable naming why). Nothing falls back
to another backend: wall time is its own backend, named in the recipe
(AGENTS.md Owner Queue, the choice rule). Backends register in BACKENDS by
name (register_backend): wall here, rapl in lassi.profilers.rapl; later
backends (AMD and NVIDIA GPUs, Tenstorrent, Cerebras) join the same table.

Sampling is lassi.profilers.power's Sampler (one daemon thread, a first
sample in start(), a last one in stop()). A counter window's energy is the
sum of the deltas of consecutive samples, each unwrapped at the counter's
range (counter_summary), so the interval must be shorter than the time the
counter takes to wrap; a power window is integrated by the trapezoid rule
(power.summarize). A failed read nulls the window's power and energy and is
never bridged (Agent Rule 1). Raw values include idle power and, for a CPU
package, every other process on that package; idle windows are P9.6's.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from lassi.core.capabilities import SUPPORTS_POWER, TAKES_DEVICE
from lassi.core.devices import DeviceSpec, parse_device
from lassi.core.record import DEVICE_KINDS, Profile
from lassi.core.registry import register
from lassi.executors.devices import HOST_ROOT
from lassi.profilers.power import (
    DEVICE_PATH,
    Sample,
    Sampler,
    TelemetryError,
    TelemetryUnavailable,
    checked_interval,
    summarize,
)

# What a backend measures on this host: a cumulative energy counter, power readings, or runtime only.
MEASURES_ENERGY = "energy"
MEASURES_POWER = "power"
MEASURES_RUNTIME = "runtime"
MEASURES = (MEASURES_ENERGY, MEASURES_POWER, MEASURES_RUNTIME)
# The label of every value the wall backend gives.
WALL_LABEL = "wall time, not energy"
# The recipe path a refusal of the backend names.
BACKEND_PATH = "profiler.backend"
# The keys a backend may take beside backend and device, as the registry admits them (EnergyProfiler.config_keys).
BACKEND_KEYS = frozenset({"interval_ms", "package"})


class TelemetrySource(Protocol):
    """The telemetry seam: what one opened backend measures on this host, and how to read it.

    `measures` is MEASURES_ENERGY (read() returns a cumulative counter in
    joules, which wraps to 0 past `range_j`, a positive number of joules),
    MEASURES_POWER (read() returns the current power in watts; `range_j` is
    None), or MEASURES_RUNTIME (nothing is read; the window's runtime is all
    there is). `label` is how every report must label the values, such as
    WALL_LABEL, or None when they are what `measures` says. record() is
    what was read, for provenance (domain names, host paths, units), with
    JSON values. A read that fails raises TelemetryError: that sample is not
    measured and its window's power and energy are null.

    A backend declares that it cannot measure by refusing to open: its
    opener raises TelemetryUnavailable naming why (a file this user cannot
    read, a domain the host lacks, a device that reports no power), before
    any window. It never returns a source that measures less than it says,
    and the profiler never falls back to another backend.
    """

    measures: str
    range_j: float | None
    label: str | None

    def read(self) -> float:
        """Return the counter in joules or the power in watts; raise TelemetryError when it cannot be read."""
        ...

    def record(self) -> dict[str, object]:
        """Return what this source reads, for the run's telemetry record (JSON values)."""
        ...


# A backend's opener: the device section as parsed, the backend's settings (its keys as given), the host root.
Opener = Callable[[DeviceSpec, Mapping[str, object], Path], TelemetrySource]


@dataclass(frozen=True)
class Backend:
    """One backend of the energy profiler: its name, the device kinds it reads, its keys, and its opener.

    `keys` are the keys it takes beside backend and device, each required
    (a subset of BACKEND_KEYS); interval_ms among them means it is sampled.
    `open` checks its settings first (ValueError naming the key, before any
    file is read), then opens the telemetry or raises TelemetryUnavailable
    naming why it cannot measure (TelemetrySource).
    """

    name: str
    kinds: frozenset[str]
    keys: frozenset[str]
    open: Opener


# The backends by name; lassi.profilers.rapl registers rapl, and later backends join here.
BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend, backends: dict[str, Backend] = BACKENDS) -> Backend:
    """Add `backend` to `backends` and return it; ValueError for a repeated name, an unknown key, or kind."""
    if backend.name in backends:
        raise ValueError(f"an energy backend named {backend.name!r} is already registered")
    unknown = sorted(backend.keys - BACKEND_KEYS)
    if unknown:
        raise ValueError(f"energy backend {backend.name!r} takes {', '.join(unknown)}, which the profiler does not")
    kinds = sorted(backend.kinds - set(DEVICE_KINDS))
    if not backend.kinds or kinds:
        raise ValueError(f"energy backend {backend.name!r} must read device kinds among {', '.join(DEVICE_KINDS)}")
    backends[backend.name] = backend
    return backend


def counter_delta(before: float, after: float, range_j: float) -> float:
    """Return the energy between two counter readings in joules: after - before, or across one wrap at range_j."""
    return after - before if after >= before else range_j - before + after


def counter_summary(samples: list[Sample], started: float, stopped: float, range_j: float) -> Profile:
    """Return the Profile of one counter window: its runtime and, when every sample was read, its energy.

    `samples` are (time, counter in joules or None) in time order, the first
    at `started` and the last at `stopped`. energy_j is the sum of the
    unwrapped deltas (counter_delta) and avg_power_w is energy_j /
    runtime_s. Both are null when any sample failed, when a reading lies
    past range_j, when fewer than two samples exist, or when the window has
    no length; runtime_s is kept. Two wraps between samples cannot be seen.
    """
    runtime = stopped - started
    values = [value for _, value in samples]
    if len(samples) < 2 or runtime <= 0 or any(value is None or value > range_j for value in values):
        return Profile(runtime_s=runtime)
    read = [value for value in values if value is not None]
    energy = sum(counter_delta(before, after, range_j) for before, after in zip(read, read[1:], strict=False))
    return Profile(runtime_s=runtime, avg_power_w=energy / runtime, energy_j=energy)


class WallSource:
    """The wall backend's source: it measures runtime only and labels its values WALL_LABEL."""

    measures: str = MEASURES_RUNTIME
    range_j: float | None = None
    label: str | None = WALL_LABEL

    def read(self) -> float:
        """Raise TelemetryError: wall time reads no telemetry."""
        raise TelemetryError("the wall backend reads no telemetry; it measures runtime only")

    def record(self) -> dict[str, object]:
        """Return the empty record: nothing is read."""
        return {}


def _open_wall(spec: DeviceSpec, settings: Mapping[str, object], root: Path) -> TelemetrySource:
    """Return the WallSource: wall time needs no telemetry and reads no file."""
    return WallSource()


register_backend(Backend(name="wall", kinds=frozenset(DEVICE_KINDS), keys=frozenset(), open=_open_wall))


class _Readings:
    """Hands a TelemetrySource's readings to power.Sampler, which samples read_w(); the values keep their unit."""

    def __init__(self, source: TelemetrySource) -> None:
        """Keep the source."""
        self._source = source

    def read_w(self) -> float:
        """Return the source's reading (joules on a counter, watts for power)."""
        return self._source.read()


def _backend(value: object) -> Backend:
    """Return the registered backend `value` names; ValueError naming profiler.backend for anything else."""
    if isinstance(value, str) and value in BACKENDS:
        return BACKENDS[value]
    given = "no value" if value is None else repr(value)
    raise ValueError(f"{BACKEND_PATH} must name one of {', '.join(sorted(BACKENDS))}, not {given}; it has no default")


def _settings(backend: Backend, given: Mapping[str, object]) -> dict[str, object]:
    """Return the backend's settings from the keys given; ValueError naming a key it does not take."""
    for key, value in given.items():
        if value is not None and key not in backend.keys:
            raise ValueError(f"profiler.{key}: backend {backend.name} takes no {key}; remove the key")
    return {key: given.get(key) for key in sorted(backend.keys)}


def _checked_source(backend: Backend, source: TelemetrySource) -> TelemetrySource:
    """Return `source` when it declares what it measures; ValueError naming the backend otherwise."""
    if source.measures not in MEASURES:
        raise ValueError(f"energy backend {backend.name} declares it measures {source.measures!r}")
    if source.measures == MEASURES_ENERGY:
        bound = source.range_j
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError(f"energy backend {backend.name} reads a counter without a range: {bound!r}")
    return source


@register("Profiler", "energy")
class EnergyProfiler:
    """Energy, power, or runtime of one device through the backend profiler.backend names (see the module docstring)."""

    name: ClassVar[str] = "energy"
    capabilities: ClassVar[frozenset[str]] = frozenset({SUPPORTS_POWER, TAKES_DEVICE})
    config_keys: ClassVar[frozenset[str]] = frozenset({"backend"}) | BACKEND_KEYS

    def __init__(
        self,
        *,
        device: Mapping[str, object],
        backend: object = None,
        interval_ms: object = None,
        package: object = None,
        source: TelemetrySource | None = None,
        root: Path = HOST_ROOT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Check the settings, open the backend's source (or take the one given), and read it once.

        In order, before any file is read: the backend (required, no
        default), the keys it takes, the device section's kind, and
        interval_ms for a sampled backend; then the backend's opener checks
        its own settings and opens. A failed first read is
        TelemetryUnavailable.
        """
        chosen = _backend(backend)
        settings = _settings(chosen, {"interval_ms": interval_ms, "package": package})
        spec = parse_device(device, DEVICE_PATH)
        if spec.kind not in chosen.kinds:
            named = " ".join([spec.kind, *(str(index) for index in spec.indices)])
            kinds = ", ".join(sorted(chosen.kinds))
            raise ValueError(f"{DEVICE_PATH} must name kind {kinds} for backend {chosen.name}, not {named}")
        interval_s = checked_interval(interval_ms) if "interval_ms" in chosen.keys else None
        opened = _checked_source(chosen, chosen.open(spec, settings, Path(root)) if source is None else source)
        self._backend = chosen
        self._source = opened
        self._clock = clock
        self._started: float | None = None
        self._sampler: Sampler | None = None
        if opened.measures == MEASURES_RUNTIME:
            return
        if interval_s is None:
            raise ValueError(f"energy backend {chosen.name} reads {opened.measures} but takes no interval_ms")
        try:
            opened.read()
        except TelemetryError as error:
            raise TelemetryUnavailable(f"the first reading of energy backend {chosen.name} failed: {error}") from None
        self._sampler = Sampler(source=_Readings(opened), interval_s=interval_s, clock=clock)

    def start(self) -> None:
        """Begin one window; ValueError when one is already open."""
        if self._sampler is not None:
            self._sampler.start()
            return
        if self._started is not None:
            raise ValueError("Profiler 'energy' was started twice; stop() it before starting it again")
        self._started = self._clock()

    def stop(self) -> Profile:
        """End the window and return its Profile; ValueError when none is open."""
        if self._sampler is None:
            if self._started is None:
                raise ValueError("Profiler 'energy' was stopped without start()")
            started, self._started = self._started, None
            return Profile(runtime_s=self._clock() - started)
        samples, started, stopped = self._sampler.stop()
        if self._source.measures == MEASURES_ENERGY:
            return counter_summary(samples, started, stopped, float(self._source.range_j or 0.0))
        return summarize(samples, started, stopped)

    def telemetry(self) -> dict[str, object]:
        """Return the run's telemetry record: profiler, backend, measures, label, and the source's record."""
        source = self._source
        head = {"profiler": self.name, "backend": self._backend.name, "measures": source.measures}
        return {**head, "label": source.label, **source.record()}
