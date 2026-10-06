"""Tests for the timing profiler (task P17.7).

Bible: Component Interfaces (Profiler), Result Record (Attempt.profile);
plans/p17-portable.md, P17.7 (the first acceptance line).

The contract these tests fix: lassi.profilers.timing.TimingProfiler,
registered as Profiler "timing", declares no capability (it reads no power
and takes no device section) and no config key. It is built as
TimingProfiler(clock=time.monotonic); start() notes the clock and stop()
returns a Profile whose runtime_s is the clock's elapsed time since start()
and whose avg_power_w and energy_j are null. The window is the one run_loop
brackets: one attempt run, sandbox setup and copy-back included, as
Attempt.run.wall_s is. stop() without start(), or start() twice, is a
ValueError, and one profiler profiles one window after another.

The clock is a PhaseClock the test moves, or the real monotonic clock; no
value in this module is a measurement.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
from profiler_fakes import PhaseClock, attribute, profilers_module

from lassi.core.record import Profile
from lassi.core.registry import DEFAULT_REGISTRY


def timing(clock: Any = None) -> Any:
    """Build a TimingProfiler on `clock`, or on its default clock when None."""
    cls = attribute(profilers_module("timing"), "TimingProfiler")
    return cls() if clock is None else cls(clock=clock)


def test_timing_reports_the_window_and_no_power() -> None:
    clock = PhaseClock(5.0)
    profiler = timing(clock)
    profiler.start()
    clock.now = 7.5
    profile = profiler.stop()
    assert isinstance(profile, Profile)
    assert profile == Profile(runtime_s=2.5), "timing reads no telemetry, so the power fields stay null"


def test_timing_profiles_one_window_after_another() -> None:
    clock = PhaseClock(10.0)
    profiler = timing(clock)
    windows = []
    for begin, end in ((10.0, 10.25), (20.0, 21.0)):
        clock.now = begin
        profiler.start()
        clock.now = end
        windows.append(profiler.stop())
    assert windows == [Profile(runtime_s=0.25), Profile(runtime_s=1.0)]


def test_timing_refuses_stop_without_start_and_a_second_start() -> None:
    profiler = timing(PhaseClock(1.0))
    with pytest.raises(ValueError):
        profiler.stop()
    profiler.start()
    with pytest.raises(ValueError):
        profiler.start()
    profiler.stop()
    with pytest.raises(ValueError):
        profiler.stop()


def test_timing_runs_on_the_monotonic_clock_by_default() -> None:
    profiler = timing()
    profiler.start()
    time.sleep(0.05)
    profile = profiler.stop()
    assert isinstance(profile.runtime_s, float) and profile.runtime_s >= 0.02, profile
    assert (profile.avg_power_w, profile.energy_j) == (None, None)


def test_timing_is_registered_without_power_or_device() -> None:
    profilers_module()
    entry = DEFAULT_REGISTRY.get("Profiler", "timing")
    assert entry.factory is attribute(profilers_module("timing"), "TimingProfiler")
    assert entry.factory.name == "timing"
    assert entry.capabilities == frozenset(), "timing declares neither supports_power nor takes_device"
    assert entry.config_keys == frozenset(), "timing takes no config key, interval_ms included"
