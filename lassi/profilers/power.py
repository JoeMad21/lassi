"""What the power profilers share (task P17.7): the source seam, the sampler, the integral, and the settings.

A power profiler (nvml, rocm_smi) reads telemetry through a PowerSource,
whose read_w() returns the current power of one GPU in watts or raises
TelemetryError for one failed read. Tests pass a fake source through the
profilers' `source` keyword; the runner builds factory(**config) and each
profiler opens its real source.

Sampling (Sampler): start() takes the first sample synchronously, then one
daemon thread named SAMPLER_THREAD samples at the interval, its waits
anchored at its own start on the monotonic clock (tick k at anchor + k *
interval; a tick missed because a read was slow is skipped, never queued),
and waiting on an Event, so stop() never waits for the next tick. stop()
sets the Event, joins the thread with no timeout, and takes the last sample
synchronously. run_loop calls stop() in a finally block (lassi.core.stages
profiled_run), so the thread is joined before the attempt's record is made,
on success and on error alike. Known cases, not complete: a source read that
never returns would block the join; each read is one sysfs read or one
library call. The thread lives here, never in lassi/core.

The integral (summarize): runtime_s is the window, stopped - started;
energy_j is the trapezoid rule over consecutive samples, in joules;
avg_power_w is energy_j / runtime_s, the time-weighted mean, in watts.
avg_power_w and energy_j are null (not measured) when any sample failed,
when fewer than two samples exist, or when the window has no length; a
failed read is never bridged or extrapolated (Agent Rule 1). Raw readings
include the GPU's idle power: idle subtraction and P9's idle windows are
not here.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import ClassVar, Protocol

from lassi.core.capabilities import SUPPORTS_POWER, TAKES_DEVICE
from lassi.core.devices import DeviceUnavailable, parse_device
from lassi.core.record import Profile
from lassi.executors.devices import HOST_ROOT

# The name of the one thread a running Sampler samples on.
SAMPLER_THREAD = "lassi-power-sampler"
# The recipe paths a power profiler's refusals name.
INTERVAL_KEY = "profiler.interval_ms"
DEVICE_PATH = "profiler.device"

# One sample: the clock's time and the power in watts, or None for a failed read.
Sample = tuple[float, float | None]


class TelemetryError(Exception):
    """One power reading that failed; the sample is recorded as not measured. Not a ValueError."""


class TelemetryUnavailable(DeviceUnavailable):
    """The power telemetry a profiler needs, missing on this host when it is built.

    It is a lassi.core.devices.DeviceUnavailable, so the runner reports it as
    the run's lack of telemetry, naming profiler.device, before any directory
    exists. It is not a ValueError, which stays the error of a refused setting.
    """


class PowerSource(Protocol):
    """The source seam: one GPU's current power."""

    def read_w(self) -> float:
        """Return the current power in watts; raise TelemetryError when it cannot be read."""
        ...


def checked_interval(value: object) -> float:
    """Return interval_ms as seconds: an integer of at least 1 (a bool is not one); ValueError naming the key."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        given = "no value" if value is None else repr(value)
        raise ValueError(f"{INTERVAL_KEY} must be an integer number of milliseconds of at least 1, not {given}")
    return value / 1000


def power_device(device: object, kind: str) -> int:
    """Return the one index of a device section of `kind`; ValueError naming profiler.device for anything else.

    The section is read as every device section is
    (lassi.core.devices.parse_device, whose DeviceSectionError is a
    ValueError); a power profiler reads one GPU, so it must name `kind` and
    exactly one index.
    """
    spec = parse_device(device, DEVICE_PATH)
    if spec.kind != kind or len(spec.indices) != 1:
        named = " ".join([spec.kind, *(str(index) for index in spec.indices)])
        raise ValueError(f"{DEVICE_PATH} must name kind {kind} and exactly one index, not {named}")
    return spec.indices[0]


def _reading(value: object) -> float | None:
    """Return a source's reading as watts, or None when it is not a finite number of at least 0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    watts = float(value)
    return watts if math.isfinite(watts) and watts >= 0 else None


class Sampler:
    """Samples a PowerSource at an interval between start() and stop(), on one thread (see the module docstring)."""

    def __init__(self, *, source: PowerSource, interval_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        """Keep the source, the interval in seconds, and the clock each sample is stamped with."""
        self._source = source
        self._interval_s = interval_s
        self._clock = clock
        self._samples: list[Sample] = []
        self._started = 0.0
        self._thread: threading.Thread | None = None
        self._done = threading.Event()
        self._error: BaseException | None = None

    def start(self) -> None:
        """Take the first sample now and start the sampler thread; ValueError when already started."""
        if self._thread is not None:
            raise ValueError("the sampler is already running; stop() it before starting it again")
        self._samples = []
        self._error = None
        self._done = threading.Event()
        self._started = self._take()
        self._thread = threading.Thread(target=self._loop, name=SAMPLER_THREAD, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[list[Sample], float, float]:
        """End the thread, join it, take the last sample; return (samples, started, stopped).

        An error other than TelemetryError raised in the thread is re-raised
        here, after the join. ValueError when the sampler is not running.
        """
        thread = self._thread
        if thread is None:
            raise ValueError("the sampler is not running; start() it first")
        self._done.set()
        thread.join()
        self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        stopped = self._take()
        return list(self._samples), self._started, stopped

    def _loop(self) -> None:
        """Sample at each tick until stop() sets the Event; keep any error other than TelemetryError for stop()."""
        anchor = time.monotonic()
        tick = 1
        try:
            while not self._done.wait(max(anchor + tick * self._interval_s - time.monotonic(), 0.0)):
                self._take()
                # A tick that passed during a slow read is skipped, never queued.
                tick = max(tick + 1, int((time.monotonic() - anchor) / self._interval_s) + 1)
        except BaseException as error:  # noqa: BLE001  (re-raised by stop() after the join)
            self._error = error

    def _take(self) -> float:
        """Read the source once, record the sample (None for a failed read), and return its time."""
        moment = self._clock()
        try:
            watts = _reading(self._source.read_w())
        except TelemetryError:
            watts = None
        self._samples.append((moment, watts))
        return moment


def summarize(samples: list[Sample], started: float, stopped: float) -> Profile:
    """Return the Profile of one window: its runtime and, when every sample was read, its trapezoid-rule power.

    `samples` come in time order, the first at `started` and the last at
    `stopped`. Power is null when any sample failed, when fewer than two
    exist, or when the window has no length; runtime_s is kept.
    """
    runtime = stopped - started
    read = [(moment, watts) for moment, watts in samples if watts is not None]
    if len(samples) < 2 or runtime <= 0 or len(read) != len(samples):
        return Profile(runtime_s=runtime)
    energy = 0.0
    for (begin, low), (end, high) in zip(read, read[1:], strict=False):
        energy += (end - begin) * (low + high) / 2
    return Profile(runtime_s=runtime, avg_power_w=energy / runtime, energy_j=energy)


class PowerProfiler:
    """The base of the power profilers: a device section, an interval, a source, and a Sampler.

    A subclass sets `name`, KIND (the device kind it reads), and
    _open_source. Building one checks the settings before any telemetry is
    touched (power_device, then checked_interval), opens the real source
    unless one is given, and takes one read, so a host without the
    telemetry is refused before the run starts: a failed read is
    TelemetryUnavailable. One profiler profiles one window after another.
    """

    name: ClassVar[str] = ""
    capabilities: ClassVar[frozenset[str]] = frozenset({SUPPORTS_POWER, TAKES_DEVICE})
    config_keys: ClassVar[frozenset[str]] = frozenset({"interval_ms"})
    KIND: ClassVar[str] = ""

    def __init__(
        self,
        *,
        device: Mapping[str, object],
        interval_ms: object = None,
        source: PowerSource | None = None,
        root: Path = HOST_ROOT,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Check the settings, open the source (or take the one given), and read it once."""
        index = power_device(device, self.KIND)
        interval_s = checked_interval(interval_ms)
        opened = self._open_source(index, Path(root)) if source is None else source
        try:
            opened.read_w()
        except TelemetryError as error:
            raise TelemetryUnavailable(
                f"the first power reading of {self.KIND} index {index} failed: {error}"
            ) from None
        self._sampler = Sampler(source=opened, interval_s=interval_s, clock=clock)

    def _open_source(self, index: int, root: Path) -> PowerSource:
        """Return the real source of GPU `index` under the host root `root`; each subclass defines it."""
        raise NotImplementedError(f"{type(self).__name__} defines no telemetry source")

    def start(self) -> None:
        """Begin one window: take the first sample and start sampling; ValueError when already started."""
        self._sampler.start()

    def stop(self) -> Profile:
        """End the window and return its Profile (summarize); ValueError when not started."""
        samples, started, stopped = self._sampler.stop()
        return summarize(samples, started, stopped)
