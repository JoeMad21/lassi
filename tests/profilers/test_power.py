"""Tests for the power profilers' shared parts (task P17.7): sampling, integration, settings, and the source seam.

Bible: Component Interfaces (Profiler; supports_power), Result Record
(Attempt.profile), Agent Rule 1, Design Principle 9; plans/p17-portable.md,
P17.7 (the first acceptance line).

The contract these tests fix (lassi.profilers.power):

- summarize(samples, started, stopped) -> Profile is a pure function over
  samples (t, watts or None) in time order, the first at `started` and the
  last at `stopped`: runtime_s is stopped - started; energy_j is the
  trapezoid rule over consecutive samples, the sum of (t[k+1] - t[k]) *
  (w[k] + w[k+1]) / 2, in joules; avg_power_w is energy_j / runtime_s, the
  time-weighted mean, in watts. avg_power_w and energy_j are null (not
  measured) when any sample failed (None), when fewer than two samples
  exist, or when the window has no length; runtime_s is kept. A failed read
  is never bridged or extrapolated (Agent Rule 1).
- Sampler(source, interval_s, clock) takes one sample synchronously in
  start(), then samples on one daemon thread named lassi-power-sampler at
  the interval; stop() ends and joins that thread, takes the last sample,
  and returns (samples, started, stopped). stop() does not wait for the
  next tick. A TelemetryError read is a failed sample (None) and sampling
  goes on; any other error in the thread is re-raised by stop() after the
  join. No sampler thread outlives stop().
- checked_interval(interval_ms) returns the interval in seconds for an
  integer of at least 1 (a bool is not one) and raises ValueError naming
  profiler.interval_ms for anything else; power_device(device, kind)
  returns the one index of a device section of `kind` and raises
  ValueError naming profiler.device for another kind or more than one index.
- TelemetryError is one failed read; TelemetryUnavailable (telemetry
  missing when a profiler is built) is a lassi.core.devices.DeviceUnavailable;
  neither is a ValueError, which stays the error of a refused setting.
- A power profiler (nvml and rocm_smi, each built with a `source`) takes
  one read when built, and a failed one is TelemetryUnavailable;
  interval_ms is required; start() twice or stop() without start() is a
  ValueError; one profiler profiles one window after another (run_loop
  calls start() and stop() around every attempt run of a run), each stop()
  returning a Profile, with no sampler thread left behind.

Every source is a FakeSource and every clock a PhaseClock or the real
monotonic clock. pynvml is made unimportable, so a profiler given a source
never reaches the vendor library. No value in this module is a measurement.
"""

from __future__ import annotations

import sys
import time
from typing import Any

import pytest
from profiler_fakes import (
    CUDA_0,
    NO_TICK_MS,
    ROCM_0,
    FakeSource,
    PhaseClock,
    attribute,
    power,
    profilers_module,
    sampler_threads,
    telemetry_error,
)

from lassi.core.devices import DeviceUnavailable
from lassi.core.record import Profile

# Each power profiler: its module, its class, and a device section of its kind.
PROFILERS = [
    pytest.param("rocm_smi", "RocmSmiProfiler", ROCM_0, id="rocm_smi"),
    pytest.param("nvml", "NvmlProfiler", CUDA_0, id="nvml"),
]
# How long a test waits for the sampler thread before it fails; a wait ends as soon as the reads arrive.
WAIT_S = 10.0


@pytest.fixture(autouse=True)
def no_vendor_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make pynvml unimportable: a profiler given a source must never reach for the vendor library."""
    monkeypatch.setitem(sys.modules, "pynvml", None)


@pytest.fixture(autouse=True)
def no_thread_left() -> Any:
    """Fail a test that starts with, or leaves behind, a live sampler thread."""
    assert sampler_threads() == [], "a sampler thread was alive before the test"
    yield
    assert sampler_threads() == [], "a sampler thread outlived the test"


def build(module: str, name: str, device: dict[str, Any], source: FakeSource, **changes: Any) -> Any:
    """Build a power profiler with `source`, the device section, and NO_TICK_MS unless `changes` say otherwise."""
    config: dict[str, Any] = {"device": dict(device), "interval_ms": NO_TICK_MS, "source": source, **changes}
    return attribute(profilers_module(module), name)(**config)


def sampler(source: FakeSource, interval_s: float) -> Any:
    """Return a Sampler over `source` at `interval_s` on the real monotonic clock."""
    return attribute(power(), "Sampler")(source=source, interval_s=interval_s, clock=time.monotonic)


# ---------------------------------------------------------------------------
# summarize: the trapezoid rule and the null rules


def test_summarize_integrates_by_the_trapezoid_rule() -> None:
    summarize = attribute(power(), "summarize")
    samples = [(10.0, 100.0), (10.5, 200.0), (11.5, 100.0), (12.0, 100.0)]
    profile = summarize(samples, 10.0, 12.0)
    assert isinstance(profile, Profile)
    # 0.5 * 150 + 1.0 * 150 + 0.5 * 100 = 275 J over 2 s: a time-weighted mean of 137.5 W, not the sample mean.
    assert profile == Profile(runtime_s=2.0, avg_power_w=137.5, energy_j=275.0)


def test_summarize_of_a_constant_power_gives_that_power() -> None:
    summarize = attribute(power(), "summarize")
    profile = summarize([(0.0, 50.0), (0.25, 50.0), (1.0, 50.0)], 0.0, 1.0)
    assert profile == Profile(runtime_s=1.0, avg_power_w=50.0, energy_j=50.0)


@pytest.mark.parametrize(
    ("samples", "started", "stopped", "runtime"),
    [
        pytest.param([(0.0, 100.0), (1.0, None), (2.0, 100.0)], 0.0, 2.0, 2.0, id="failed-middle"),
        pytest.param([(0.0, None), (1.0, 100.0)], 0.0, 1.0, 1.0, id="failed-first"),
        pytest.param([(0.0, 100.0), (1.0, None)], 0.0, 1.0, 1.0, id="failed-last"),
    ],
)
def test_summarize_gives_null_power_for_a_failed_sample(
    samples: list[tuple[float, float | None]], started: float, stopped: float, runtime: float
) -> None:
    profile = attribute(power(), "summarize")(samples, started, stopped)
    assert profile == Profile(runtime_s=runtime), "a failed read is never bridged; the power is not measured"


@pytest.mark.parametrize(
    ("samples", "started", "stopped", "runtime"),
    [
        pytest.param([(3.0, 100.0)], 3.0, 3.0, 0.0, id="one-sample"),
        pytest.param([], 1.0, 2.0, 1.0, id="no-sample"),
        pytest.param([(5.0, 100.0), (5.0, 120.0)], 5.0, 5.0, 0.0, id="empty-window"),
    ],
)
def test_summarize_gives_null_power_for_one_sample_or_an_empty_window(
    samples: list[tuple[float, float | None]], started: float, stopped: float, runtime: float
) -> None:
    profile = attribute(power(), "summarize")(samples, started, stopped)
    assert profile == Profile(runtime_s=runtime)


def test_summarize_keeps_runtime_when_power_is_null() -> None:
    profile = attribute(power(), "summarize")([(7.0, None), (9.5, None)], 7.0, 9.5)
    assert (profile.runtime_s, profile.avg_power_w, profile.energy_j) == (2.5, None, None)


# ---------------------------------------------------------------------------
# The Sampler: a start sample, samples on one thread at the interval, a stop sample, and the join


def test_sampler_takes_a_start_and_a_stop_sample_and_samples_between() -> None:
    source = FakeSource(default=40.0, notify_after=5)
    running = sampler(source, 0.001)
    running.start()
    assert source.reads >= 1, "start() takes the first sample before it returns"
    assert source.reached.wait(WAIT_S), "the sampler thread took no sample at a 1 ms interval"
    samples, started, stopped = running.stop()
    reads = source.reads
    assert len(samples) == reads >= 6, "the start sample, at least four from the thread, and the stop sample"
    assert samples[0][0] == started and samples[-1][0] == stopped
    times = [moment for moment, _ in samples]
    assert times == sorted(times), "samples come in time order"
    assert all(value == 40.0 for _, value in samples)
    time.sleep(0.05)
    assert source.reads == reads, "nothing reads the source after stop() returns"


def test_sampler_thread_is_joined_by_stop() -> None:
    running = sampler(FakeSource(default=10.0), 0.01)
    running.start()
    threads = sampler_threads()
    assert len(threads) == 1, "one sampler thread runs between start() and stop()"
    assert threads[0].daemon, "the sampler thread is a daemon thread"
    running.stop()
    assert sampler_threads() == [], "stop() joins the sampler thread"


def test_sampler_samples_at_the_interval_and_stop_does_not_wait_for_the_next_tick() -> None:
    source = FakeSource(default=10.0)
    running = sampler(source, NO_TICK_MS / 1000)
    running.start()
    time.sleep(0.05)
    begun = time.monotonic()
    samples, _, _ = running.stop()
    assert time.monotonic() - begun < 5.0, "stop() ends the wait for the next tick"
    assert source.reads == len(samples) == 2, "no tick of a 60 s interval falls inside the window"


def test_sampler_records_a_failed_read_as_a_failed_sample_and_samples_on() -> None:
    source = FakeSource([10.0, 10.0, telemetry_error()], default=10.0, notify_after=6)
    running = sampler(source, 0.001)
    running.start()
    assert source.reached.wait(WAIT_S), "the sampler thread stopped after a failed read"
    samples, _, _ = running.stop()
    values = [value for _, value in samples]
    assert values[2] is None, "the failed read is a failed sample, never a value"
    assert all(value == 10.0 for value in values[3:]) and len(values) >= 6, "sampling goes on after it"


def test_a_failed_start_sample_is_a_failed_sample() -> None:
    source = FakeSource([telemetry_error(), 30.0])
    running = sampler(source, NO_TICK_MS / 1000)
    running.start()
    samples, _, _ = running.stop()
    assert [value for _, value in samples] == [None, 30.0]


def test_sampler_stop_joins_after_a_source_error_and_reraises_it() -> None:
    source = FakeSource([5.0, 5.0, RuntimeError("SYNTHETIC: the source broke")], default=5.0, notify_after=3)
    running = sampler(source, 0.001)
    running.start()
    assert source.reached.wait(WAIT_S), "the sampler thread never read the broken source"
    with pytest.raises(RuntimeError, match="SYNTHETIC: the source broke"):
        running.stop()
    assert sampler_threads() == [], "the thread is joined before the error is re-raised"


# ---------------------------------------------------------------------------
# Settings: interval_ms and the device section


def test_interval_ms_converts_to_seconds() -> None:
    checked = attribute(power(), "checked_interval")
    assert checked(10) == pytest.approx(0.01)
    assert checked(1) == pytest.approx(0.001)


@pytest.mark.parametrize(
    "value", [0, -1, 2.5, True, None, "10"], ids=["zero", "negative", "float", "bool", "none", "str"]
)
def test_interval_ms_must_be_an_integer_of_at_least_one(value: object) -> None:
    with pytest.raises(ValueError, match="profiler.interval_ms"):
        attribute(power(), "checked_interval")(value)


def test_power_device_returns_the_one_index_of_its_kind() -> None:
    device = attribute(power(), "power_device")
    assert device({"kind": "rocm", "indices": [1]}, "rocm") == 1
    assert device({"kind": "cuda", "indices": [0]}, "cuda") == 0


@pytest.mark.parametrize(
    ("section", "kind"),
    [
        pytest.param({"kind": "cuda", "indices": [0]}, "rocm", id="another-kind"),
        pytest.param({"kind": "rocm", "indices": [0, 1]}, "rocm", id="two-indices"),
        pytest.param({"kind": "cpu"}, "rocm", id="cpu"),
    ],
)
def test_power_device_needs_its_kind_and_one_index(section: dict[str, Any], kind: str) -> None:
    with pytest.raises(ValueError, match="profiler.device"):
        attribute(power(), "power_device")(section, kind)


def test_the_telemetry_errors_are_not_setting_errors() -> None:
    module = power()
    unavailable = attribute(module, "TelemetryUnavailable")
    failed = attribute(module, "TelemetryError")
    assert attribute(module, "PowerSource") is not None
    assert issubclass(unavailable, DeviceUnavailable), "the runner reports it as it reports a missing device"
    assert not issubclass(unavailable, ValueError) and not issubclass(failed, ValueError)


# ---------------------------------------------------------------------------
# A power profiler built with a source


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_power_profiler_reads_its_source_once_when_built(module: str, name: str, device: dict[str, Any]) -> None:
    source = FakeSource([10.0])
    build(module, name, device, source)
    assert source.reads == 1, "the build reads the telemetry once, so a run without it is refused before it starts"


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_failed_first_read_is_telemetry_unavailable(module: str, name: str, device: dict[str, Any]) -> None:
    unavailable = attribute(power(), "TelemetryUnavailable")
    with pytest.raises(unavailable):
        build(module, name, device, FakeSource([telemetry_error("SYNTHETIC: no power reading here")]))


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_interval_ms_is_required_and_checked_at_build(module: str, name: str, device: dict[str, Any]) -> None:
    cls = attribute(profilers_module(module), name)
    with pytest.raises(ValueError, match="profiler.interval_ms"):
        cls(device=dict(device), source=FakeSource(default=1.0))
    with pytest.raises(ValueError, match="profiler.interval_ms"):
        build(module, name, device, FakeSource(default=1.0), interval_ms=0)


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_device_of_another_kind_or_two_indices_is_refused_at_build(
    module: str, name: str, device: dict[str, Any]
) -> None:
    other = CUDA_0 if device["kind"] == "rocm" else ROCM_0
    with pytest.raises(ValueError, match="profiler.device"):
        build(module, name, other, FakeSource(default=1.0))
    with pytest.raises(ValueError, match="profiler.device"):
        build(module, name, {"kind": device["kind"], "indices": [0, 1]}, FakeSource(default=1.0))


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_power_profiler_reports_runtime_power_and_energy_over_its_window(
    module: str, name: str, device: dict[str, Any]
) -> None:
    clock = PhaseClock(100.0)
    profiler = build(module, name, device, FakeSource([1.0, 40.0, 60.0]), clock=clock)
    profiler.start()
    clock.now = 102.0
    profile = profiler.stop()
    assert profile == Profile(runtime_s=2.0, avg_power_w=50.0, energy_j=100.0)


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_failed_read_in_the_window_leaves_power_null_and_keeps_runtime(
    module: str, name: str, device: dict[str, Any]
) -> None:
    clock = PhaseClock(100.0)
    profiler = build(module, name, device, FakeSource([1.0, 40.0, telemetry_error()]), clock=clock)
    profiler.start()
    clock.now = 102.0
    assert profiler.stop() == Profile(runtime_s=2.0)


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_profiler_refuses_stop_without_start_and_a_second_start(
    module: str, name: str, device: dict[str, Any]
) -> None:
    profiler = build(module, name, device, FakeSource(default=5.0), clock=PhaseClock(1.0))
    with pytest.raises(ValueError):
        profiler.stop()
    profiler.start()
    with pytest.raises(ValueError):
        profiler.start()
    assert isinstance(profiler.stop(), Profile)
    with pytest.raises(ValueError):
        profiler.stop()


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_profiler_profiles_one_window_after_another(module: str, name: str, device: dict[str, Any]) -> None:
    clock = PhaseClock(100.0)
    profiler = build(module, name, device, FakeSource([1.0, 40.0, 60.0, 10.0, 30.0]), clock=clock)
    profiler.start()
    clock.now = 102.0
    first = profiler.stop()
    assert sampler_threads() == []
    clock.now = 200.0
    profiler.start()
    clock.now = 201.0
    second = profiler.stop()
    assert (first, second) == (
        Profile(runtime_s=2.0, avg_power_w=50.0, energy_j=100.0),
        Profile(runtime_s=1.0, avg_power_w=20.0, energy_j=20.0),
    )


@pytest.mark.parametrize(("module", "name", "device"), PROFILERS)
def test_a_power_profiler_on_the_real_clock_samples_at_its_interval(
    module: str, name: str, device: dict[str, Any]
) -> None:
    source = FakeSource(default=100.0, notify_after=6)
    profiler = build(module, name, device, source, interval_ms=1, clock=time.monotonic)
    profiler.start()
    assert source.reached.wait(WAIT_S), "the profiler took no sample between start() and stop()"
    time.sleep(0.05)
    profile = profiler.stop()
    assert profile.runtime_s is not None and profile.runtime_s > 0
    assert profile.avg_power_w == pytest.approx(100.0)
    assert profile.energy_j == pytest.approx(100.0 * profile.runtime_s)
    assert source.reads >= 7, "the build read, the start sample, at least four ticks, and the stop sample"
