"""The timing profiler, registered as Profiler "timing" (task P17.7; bible Component Interfaces, Profiler).

It reads no telemetry and takes no device section, so it declares no
capability and no config key, and its Profile has runtime_s only (power
null). runtime_s is the clock's elapsed time between start() and stop(),
which run_loop calls around one attempt run (lassi.core.stages
profiled_run). Like Attempt.run.wall_s, that window covers the whole
sandbox command, setup and copy-back included, so it is not the program's
own time. The clock is a keyword (time.monotonic by default) so tests can
move it by hand.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import ClassVar

from lassi.core.record import Profile
from lassi.core.registry import register


@register("Profiler", "timing")
class TimingProfiler:
    """Times one window after another on a monotonic clock; reports runtime_s and no power."""

    name: ClassVar[str] = "timing"
    capabilities: ClassVar[frozenset[str]] = frozenset()
    config_keys: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        """Keep the clock; no window is open."""
        self._clock = clock
        self._started: float | None = None

    def start(self) -> None:
        """Open a window at the clock's time now; ValueError when one is already open."""
        if self._started is not None:
            raise ValueError("Profiler 'timing' was started twice; stop() it before starting it again")
        self._started = self._clock()

    def stop(self) -> Profile:
        """Close the window and return Profile(runtime_s=its length); ValueError when none is open."""
        if self._started is None:
            raise ValueError("Profiler 'timing' was stopped without start()")
        started, self._started = self._started, None
        return Profile(runtime_s=self._clock() - started)
