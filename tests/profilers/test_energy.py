"""Tests for the energy profiler and its telemetry seam (task P9.1; OQ-088): one interface, backends behind it.

Bible: Component Interfaces (Profiler; supports_power; contract rules),
Result Record (Profiles), Project Recipes (Notes, Profilers), Design
Principle 9, Agent Rules 1 and 2; plans/p9-ee.md, P9.1 and the planning
decision "One energy interface".

The contract these tests fix (lassi.profilers.energy):

- EnergyProfiler, registered as Profiler "energy", declares supports_power
  and takes_device and takes the config keys backend, interval_ms, and
  package; it is built with keyword-only device, backend, interval_ms,
  package, source, root, and clock. profiler.backend is required, with no
  default, and names a registered backend (BACKENDS: at least rapl and
  wall); a missing or unknown one is a ValueError naming profiler.backend.
  A key the named backend does not take is a ValueError naming it; nothing
  is ignored.
- The seam, TelemetrySource: `measures` (energy, power, or runtime),
  `range_j` (an energy counter's range in joules, None otherwise), `label`
  (how a report must label the values, or None), read() (joules on the
  counter, or watts), and record() (what was read, for provenance). A
  backend that cannot measure refuses when opened (TelemetryUnavailable
  naming why); there is no fallback to another backend.
- A counter window: counter_delta(before, after, range_j) is after -
  before, or range_j - before + after when the counter wrapped;
  counter_summary sums the deltas of consecutive samples. Energy and power
  are null when any sample failed (never bridged), when fewer than two
  samples exist, when the window has no length, or when a reading lies
  past the range; runtime_s is kept.
- A power source is integrated as lassi.profilers.power summarize does.
- Backend wall measures runtime only: its Profile has runtime_s and null
  power, and its telemetry record is labeled "wall time, not energy".
- telemetry() returns the run's telemetry record: profiler, backend,
  measures, label, and the source's own record.

Every source is a fake and every clock a PhaseClock; no value in this
module is a measurement.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from profiler_fakes import NO_TICK_MS, PhaseClock, attribute, profilers_module, sampler_threads, telemetry_error

from lassi.core.devices import DeviceSpec
from lassi.core.record import Profile
from lassi.core.registry import DEFAULT_REGISTRY

CPU = {"kind": "cpu"}
ROCM_0 = {"kind": "rocm", "indices": [0]}
WALL_LABEL = "wall time, not energy"
# A SYNTHETIC counter range in joules: 1000 J, as a max_energy_range_uj of 1000000000.
RANGE_J = 1000.0


def energy() -> ModuleType:
    """Return lassi.profilers.energy, failing clearly while task P9.1 has not added it."""
    profilers_module()
    return profilers_module("energy")


def unavailable() -> type[BaseException]:
    """Return lassi.profilers.power.TelemetryUnavailable."""
    return attribute(profilers_module("power"), "TelemetryUnavailable")


class FakeTelemetry:
    """A TelemetrySource that answers from a script: each read takes the next item, a value or an exception."""

    def __init__(
        self,
        script: Iterable[float | BaseException] = (),
        *,
        measures: str = "energy",
        range_j: float | None = RANGE_J,
        label: str | None = None,
        default: float | None = None,
    ) -> None:
        """Keep the script, what the source declares it measures, and the reading after the script runs out."""
        self.script = list(script)
        self.measures = measures
        self.range_j = range_j
        self.label = label
        self.default = default
        self.reads = 0
        self._lock = threading.Lock()

    def read(self) -> float:
        """Return the next scripted reading (joules on the counter, or watts), or raise the scripted exception."""
        with self._lock:
            self.reads += 1
            item = self.script.pop(0) if self.script else self.default
        if item is None:
            raise AssertionError("the source was read more often than the test scripted")
        if isinstance(item, BaseException):
            raise item
        return item

    def record(self) -> dict[str, object]:
        """Return the SYNTHETIC provenance record of this source."""
        return {"source": "SYNTHETIC fake"}


def profiler(backend: str | None = "rapl", device: Mapping[str, Any] = CPU, **changes: Any) -> Any:
    """Build an EnergyProfiler with `changes` (a source, a clock, settings); rapl gets package 0 at NO_TICK_MS."""
    config: dict[str, Any] = {"device": dict(device), "backend": backend}
    if backend == "rapl":
        config.update(interval_ms=NO_TICK_MS, package=0)
    config.update(changes)
    return attribute(energy(), "EnergyProfiler")(**config)


# ---------------------------------------------------------------------------
# Registration and the required backend


def test_energy_is_registered_with_supports_power_takes_device_and_its_keys() -> None:
    entry = DEFAULT_REGISTRY.get("Profiler", "energy")
    cls = attribute(energy(), "EnergyProfiler")
    assert entry.factory is cls and cls.name == "energy"
    assert entry.capabilities == frozenset({"supports_power", "takes_device"})
    assert entry.config_keys == frozenset({"backend", "interval_ms", "package"}), "idle_window_s stays P9.6's"
    parameters = inspect.signature(cls).parameters
    for key in ("device", "backend", "interval_ms", "package", "source", "root", "clock"):
        assert parameters[key].kind is inspect.Parameter.KEYWORD_ONLY, key


def test_the_backends_include_rapl_and_wall() -> None:
    backends = attribute(energy(), "BACKENDS")
    assert {"rapl", "wall"} <= set(backends), sorted(backends)
    assert backends["rapl"].kinds == frozenset({"cpu"})
    assert backends["rapl"].keys == frozenset({"interval_ms", "package"})
    assert backends["wall"].keys == frozenset()


@pytest.mark.parametrize("backend", [None, "perf", "", 3], ids=["missing", "unknown", "empty", "number"])
def test_the_backend_is_required_with_no_default(backend: Any) -> None:
    with pytest.raises(ValueError, match="profiler.backend") as caught:
        profiler(backend, source=FakeTelemetry(default=1.0))
    assert "rapl" in str(caught.value) and "wall" in str(caught.value), "the refusal lists the backends"


@pytest.mark.parametrize(
    ("changes", "key"),
    [({"interval_ms": 10}, "profiler.interval_ms"), ({"package": 0}, "profiler.package")],
    ids=["interval", "package"],
)
def test_a_key_the_backend_does_not_take_is_refused(changes: dict[str, Any], key: str) -> None:
    with pytest.raises(ValueError, match=key) as caught:
        profiler("wall", **changes)
    assert "wall" in str(caught.value), "the refusal names the backend"


def test_a_device_kind_the_backend_does_not_read_is_refused() -> None:
    with pytest.raises(ValueError, match="profiler.device") as caught:
        profiler("rapl", device=ROCM_0, source=FakeTelemetry(default=1.0))
    assert "cpu" in str(caught.value) and "rapl" in str(caught.value)


def test_rapl_needs_its_interval() -> None:
    with pytest.raises(ValueError, match="profiler.interval_ms"):
        profiler("rapl", interval_ms=None, source=FakeTelemetry(default=1.0))


# ---------------------------------------------------------------------------
# The counter arithmetic


def test_counter_delta_without_and_with_a_wrap() -> None:
    delta = attribute(energy(), "counter_delta")
    assert delta(100.0, 250.0, RANGE_J) == 150.0
    assert delta(990.0, 15.0, RANGE_J) == 25.0, "a wrap adds the range: (1000 - 990) + 15"
    assert delta(400.0, 400.0, RANGE_J) == 0.0


def test_counter_summary_sums_the_deltas_across_a_wrap() -> None:
    summary = attribute(energy(), "counter_summary")
    samples = [(0.0, 900.0), (1.0, 980.0), (2.0, 30.0), (4.0, 70.0)]
    assert summary(samples, 0.0, 4.0, RANGE_J) == Profile(runtime_s=4.0, avg_power_w=42.5, energy_j=170.0)


@pytest.mark.parametrize(
    "samples",
    [
        [(0.0, 100.0), (1.0, None), (2.0, 300.0)],
        [(0.0, None), (2.0, 300.0)],
        [(0.0, 100.0), (2.0, None)],
    ],
    ids=["middle", "first", "last"],
)
def test_a_failed_sample_nulls_the_window_and_is_never_bridged(samples: list[tuple[float, float | None]]) -> None:
    summary = attribute(energy(), "counter_summary")
    assert summary(samples, 0.0, 2.0, RANGE_J) == Profile(runtime_s=2.0)


def test_counter_summary_needs_two_samples_a_window_and_readings_inside_the_range() -> None:
    summary = attribute(energy(), "counter_summary")
    assert summary([(0.0, 100.0)], 0.0, 0.0, RANGE_J) == Profile(runtime_s=0.0)
    assert summary([(1.0, 100.0), (1.0, 200.0)], 1.0, 1.0, RANGE_J) == Profile(runtime_s=0.0)
    assert summary([(0.0, 100.0), (1.0, RANGE_J + 1.0)], 0.0, 1.0, RANGE_J) == Profile(runtime_s=1.0)


# ---------------------------------------------------------------------------
# The profiler over the seam: counters, power, and runtime only


def test_a_counter_source_gives_the_window_energy_and_mean_power() -> None:
    clock = PhaseClock(10.0)
    source = FakeTelemetry([500.0, 500.0, 620.0])
    built = profiler("rapl", source=source, clock=clock)
    built.start()
    clock.now = 13.0
    assert built.stop() == Profile(runtime_s=3.0, avg_power_w=40.0, energy_j=120.0)
    assert source.reads == 3, "one read when built, one at start, one at stop"
    assert sampler_threads() == []


def test_a_counter_that_wraps_in_the_window_is_unwrapped() -> None:
    clock = PhaseClock(0.0)
    built = profiler("rapl", source=FakeTelemetry([990.0, 990.0, 15.0]), clock=clock)
    built.start()
    clock.now = 5.0
    assert built.stop() == Profile(runtime_s=5.0, avg_power_w=5.0, energy_j=25.0)


def test_a_failed_read_in_the_window_nulls_its_energy_and_keeps_runtime() -> None:
    clock = PhaseClock(0.0)
    built = profiler("rapl", source=FakeTelemetry([100.0, 100.0, telemetry_error()]), clock=clock)
    built.start()
    clock.now = 2.0
    assert built.stop() == Profile(runtime_s=2.0), "a failed read is a null value, never bridged"


def test_a_failed_first_read_is_telemetry_unavailable() -> None:
    with pytest.raises(unavailable()):
        profiler("rapl", source=FakeTelemetry([telemetry_error()]))


def test_a_counter_source_without_a_range_is_refused() -> None:
    with pytest.raises((ValueError, unavailable())):
        profiler("rapl", source=FakeTelemetry(default=1.0, range_j=None))


def test_a_power_source_is_integrated_by_the_trapezoid_rule() -> None:
    clock = PhaseClock(0.0)
    source = FakeTelemetry([100.0, 100.0, 140.0], measures="power", range_j=None)
    built = profiler("rapl", source=source, clock=clock)
    built.start()
    clock.now = 2.0
    assert built.stop() == Profile(runtime_s=2.0, avg_power_w=120.0, energy_j=240.0)


def test_one_profiler_profiles_one_window_after_another() -> None:
    clock = PhaseClock(0.0)
    built = profiler("rapl", source=FakeTelemetry([0.0, 0.0, 10.0, 10.0, 40.0]), clock=clock)
    built.start()
    clock.now = 1.0
    first = built.stop()
    built.start()
    clock.now = 3.0
    assert (first, built.stop()) == (
        Profile(runtime_s=1.0, avg_power_w=10.0, energy_j=10.0),
        Profile(runtime_s=2.0, avg_power_w=15.0, energy_j=30.0),
    )


@pytest.mark.parametrize("backend", ["rapl", "wall"])
def test_start_twice_and_stop_without_start_are_value_errors(backend: str) -> None:
    extra = {"source": FakeTelemetry(default=1.0)} if backend == "rapl" else {}
    built = profiler(backend, **extra)
    with pytest.raises(ValueError):
        built.stop()
    built.start()
    with pytest.raises(ValueError):
        built.start()
    built.stop()
    assert sampler_threads() == []


# ---------------------------------------------------------------------------
# wall: runtime only, labeled


@pytest.mark.parametrize("device", [CPU, ROCM_0], ids=["cpu", "rocm"])
def test_wall_measures_runtime_only_and_is_labeled_not_energy(device: dict[str, Any]) -> None:
    clock = PhaseClock(5.0)
    built = profiler("wall", device=device, clock=clock)
    built.start()
    clock.now = 7.5
    assert built.stop() == Profile(runtime_s=2.5), "no power and no energy"
    record = built.telemetry()
    assert record["profiler"] == "energy" and record["backend"] == "wall"
    assert record["measures"] == "runtime" and record["label"] == WALL_LABEL


def test_telemetry_names_the_backend_what_it_measures_and_the_source_record() -> None:
    record = profiler("rapl", source=FakeTelemetry(default=1.0)).telemetry()
    assert record == {
        "profiler": "energy",
        "backend": "rapl",
        "measures": "energy",
        "label": None,
        "source": "SYNTHETIC fake",
    }


# ---------------------------------------------------------------------------
# Backends behind the seam: registration and a backend that cannot measure


@dataclass(frozen=True)
class Opened:
    """What a fake backend's opener was given."""

    spec: DeviceSpec
    settings: dict[str, object]
    root: Path


def test_a_backend_that_cannot_measure_refuses_when_built_and_nothing_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = energy()
    reason = "SYNTHETIC: the fabric simulator reports no power"

    def refuse(spec: DeviceSpec, settings: Mapping[str, object], root: Path) -> Any:
        raise unavailable()(reason)

    backend = attribute(module, "Backend")(name="unmeasured", kinds=frozenset({"cpu"}), keys=frozenset(), open=refuse)
    monkeypatch.setitem(module.BACKENDS, "unmeasured", backend)
    with pytest.raises(unavailable(), match="fabric simulator reports no power"):
        profiler("unmeasured")


def test_a_backend_is_opened_with_the_device_its_settings_and_the_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = energy()
    seen: list[Opened] = []

    def opened(spec: DeviceSpec, settings: Mapping[str, object], root: Path) -> Any:
        seen.append(Opened(spec, dict(settings), Path(root)))
        return FakeTelemetry(default=1.0, measures="power", range_j=None)

    keys = frozenset({"interval_ms", "package"})
    backend = attribute(module, "Backend")(name="opened", kinds=frozenset({"cpu"}), keys=keys, open=opened)
    monkeypatch.setitem(module.BACKENDS, "opened", backend)
    profiler("opened", interval_ms=10, package=1, root=tmp_path)
    assert seen == [Opened(DeviceSpec("cpu", ()), {"interval_ms": 10, "package": 1}, tmp_path)]


def test_register_backend_refuses_a_repeated_name_and_a_key_the_profiler_does_not_take() -> None:
    module = energy()
    backend = attribute(module, "Backend")
    register = attribute(module, "register_backend")
    table: dict[str, Any] = {}

    def never(spec: DeviceSpec, settings: Mapping[str, object], root: Path) -> Any:
        raise AssertionError("not opened")

    register(backend(name="one", kinds=frozenset({"cpu"}), keys=frozenset(), open=never), table)
    with pytest.raises(ValueError, match="one"):
        register(backend(name="one", kinds=frozenset({"cpu"}), keys=frozenset(), open=never), table)
    with pytest.raises(ValueError, match="idle_window_s"):
        register(backend(name="two", kinds=frozenset({"cpu"}), keys=frozenset({"idle_window_s"}), open=never), table)
    with pytest.raises(ValueError, match="kind"):
        register(backend(name="three", kinds=frozenset({"tpu"}), keys=frozenset(), open=never), table)


def test_the_seam_docstring_says_how_a_backend_declares_it_cannot_measure() -> None:
    doc = attribute(energy(), "TelemetrySource").__doc__ or ""
    assert "TelemetryUnavailable" in doc and "cannot measure" in doc, doc
