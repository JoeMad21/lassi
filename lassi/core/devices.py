"""The device layer's device-neutral parts: the device section, the probe seam, and the device record (task P17.2).

A recipe names every device a component uses explicitly, with no default
(bible Project Recipes, Notes; Agent Rule 10). A device section is the
mapping `{kind, indices}` under the key `device` of a component's own recipe
section, for a component that declares the capability `takes_device`
(lassi.core.capabilities TAKES_DEVICE). parse_device reads one section into a
DeviceSpec; the recipe loader calls it at load, and the train loader (task
P17.8) reuses it.

Before any component is built or any directory exists, the runner probes
each named device with probe_device and records the DeviceRecord it returns
in provenance.json and every trial's provenance (bible Result Record). A
probe is a DeviceProbe registered by kind in DEFAULT_PROBES; the host probes
live in lassi.executors.devices, since device files are read only in
lassi/executors and lassi/profilers (Design Principle 9). A probe reads file
metadata (stat, access) and procfs or sysfs text only: it never opens a
device node, starts no process, and imports no framework (OQ-002). A
framework's build metadata (FrameworkBuild) comes from the component class
beside its adapter, never from this module.

Nothing here reads a file or imports an ML framework, and nothing falls back
to another device: probe_device checks only the kind the recipe names.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from lassi.core.record import DEVICE_KINDS, DeviceRecord

# The keys a device section may hold.
_SECTION_KEYS = ("indices", "kind")


class DeviceSectionError(ValueError):
    """A device section that does not have the one shape; the message names the dotted key."""


class DeviceUnavailable(Exception):
    """A named device this host cannot provide; the message says why.

    It is not a ValueError, so no handler written for a component's own
    errors catches it by accident.
    """


@dataclass(frozen=True)
class DeviceSpec:
    """One parsed device section: the kind and the indices as written (() for cpu)."""

    kind: str
    indices: tuple[int, ...]


@dataclass(frozen=True)
class FrameworkBuild:
    """The build metadata of the framework a component runs on.

    For PyTorch: name "torch", version torch.__version__, cuda
    torch.version.cuda, and hip torch.version.hip (None when the build has
    none). It is read from build metadata only, never by a call that
    initializes a device runtime.
    """

    name: str
    version: str
    cuda: str | None
    hip: str | None


@dataclass(frozen=True)
class HostFacts:
    """What a host probe read for one device kind; None means not read.

    count is the number of devices of the kind the host has (None only when
    the host reports no count), name the device's name, memory_bytes its
    memory, driver the driver version, and runtime the device runtime's
    version. For a kind whose devices are numbered nodes (tenstorrent; task
    P17.38), by_index maps the number of each device this user can use to
    its description, and refused the number of each other device to why it
    cannot be used; both are None for every other kind.
    """

    count: int | None
    name: str | None
    memory_bytes: int | None
    driver: str | None
    runtime: str | None
    by_index: Mapping[int, str] | None = None
    refused: Mapping[int, str] | None = None


class DeviceProbe(Protocol):
    """A host probe for one device kind.

    probe() returns the host's facts for the kind, or raises
    DeviceUnavailable with the reason and nothing else. It reads file
    metadata and procfs or sysfs text only: it never opens a device node,
    starts no process, and imports no framework.
    """

    kind: str

    def probe(self) -> HostFacts:
        """Return the host's facts for this kind, or raise DeviceUnavailable."""
        ...


# The probes by kind that the runner uses when RunOptions.probes is None; lassi.executors registers its host probes.
DEFAULT_PROBES: dict[str, DeviceProbe] = {}
# The kind of a Tenstorrent card, and the framework builds that can reach one: the tt-xla wheel's distribution
# (toolchains/tt-xla.pin; task P17.38).
TENSTORRENT = "tenstorrent"
TENSTORRENT_FRAMEWORKS = ("pjrt-plugin-tt",)


def parse_device(value: Any, path: str) -> DeviceSpec:
    """Return the DeviceSpec of the device section `value` found at the dotted recipe path `path`.

    The section is a mapping with a required kind, one of DEVICE_KINDS as
    written (no case folding), and indices: for cuda, rocm, and tenstorrent
    a non-empty list of distinct integers of at least 0 (a bool is not one;
    for tenstorrent the cards' /dev/tenstorrent numbers), kept in the
    order written; for cpu no indices key at all. Any other key, and a value
    that is not a mapping, is refused. Every refusal is a DeviceSectionError
    naming the dotted key.
    """
    if not isinstance(value, dict):
        raise DeviceSectionError(f"{path} must be a mapping {{kind, indices}}, not {value!r}")
    for key in value:
        if key not in _SECTION_KEYS:
            raise DeviceSectionError(f"unknown key {path}.{key}; allowed here: {', '.join(_SECTION_KEYS)}")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in DEVICE_KINDS:
        given = "no value" if kind is None else repr(kind)
        raise DeviceSectionError(f"{path}.kind must be one of {', '.join(DEVICE_KINDS)}, not {given}")
    if kind == "cpu":
        if "indices" in value:
            raise DeviceSectionError(f"{path}.indices: kind cpu takes no indices; remove the key")
        return DeviceSpec(kind=kind, indices=())
    return DeviceSpec(kind=kind, indices=_indices(value.get("indices"), f"{path}.indices", kind))


def _indices(value: Any, path: str, kind: str) -> tuple[int, ...]:
    """Return the indices of a GPU kind as written; refuse anything but distinct integers of at least 0."""
    if not isinstance(value, list) or not value:
        raise DeviceSectionError(f"{path} must be a non-empty list of device indices for kind {kind}, not {value!r}")
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise DeviceSectionError(f"{path} must hold integers of at least 0, not {item!r}")
    repeated = sorted({item for item in value if value.count(item) > 1})
    if repeated:
        raise DeviceSectionError(f"{path} names index {repeated[0]} more than once; name each device once")
    return tuple(value)


def register_probe(probe: DeviceProbe, probes: dict[str, DeviceProbe] = DEFAULT_PROBES) -> None:
    """Register `probe` in `probes` under its kind; ValueError for a kind outside DEVICE_KINDS or one already there."""
    kind = probe.kind
    if kind not in DEVICE_KINDS:
        raise ValueError(f"a device probe's kind must be one of {', '.join(DEVICE_KINDS)}, not {kind!r}")
    if kind in probes:
        raise ValueError(f"a device probe of kind {kind!r} is already registered")
    probes[kind] = probe


def probe_device(
    key: str, spec: DeviceSpec, framework: FrameworkBuild | None, probes: Mapping[str, DeviceProbe]
) -> DeviceRecord:
    """Return the DeviceRecord of the device section at `key`, or raise DeviceUnavailable; never another kind.

    In order: the kind needs a probe in `probes`; the framework build must fit
    the kind before any probe runs (cuda needs a CUDA build without HIP; rocm
    a HIP build, since ROCm PyTorch reaches its GPUs through the cuda device
    type; tenstorrent one of TENSTORRENT_FRAMEWORKS; cpu fits any build, and
    None means no framework); the probe runs; and every index must be below
    the host's count (a count of None refuses any index), or, when the probe
    gives by_index, be one of its keys, the record's name then joining their
    descriptions in the order written. The record's runtime is None for cpu
    and tenstorrent, and for a GPU kind the build's CUDA or HIP version, or
    the probe's runtime without a framework.
    """
    probe = probes.get(spec.kind)
    if probe is None:
        raise DeviceUnavailable(f"no {spec.kind} device probe is registered, so this host cannot provide one")
    _check_framework(spec.kind, framework)
    facts = probe.probe()
    if facts.by_index is None:
        _check_counted(spec, facts)
        name = facts.name
    else:
        name = _indexed_name(spec, facts)
    return DeviceRecord(
        key=key,
        kind=spec.kind,
        indices=list(spec.indices),
        name=name,
        count=facts.count,
        memory_bytes=facts.memory_bytes,
        driver=facts.driver,
        runtime=_runtime(spec.kind, framework, facts),
        framework=None if framework is None else framework.name,
        framework_version=None if framework is None else framework.version,
    )


def _check_counted(spec: DeviceSpec, facts: HostFacts) -> None:
    """Raise DeviceUnavailable for an index at or past the host's count of the kind (any index when it is None)."""
    for index in spec.indices:
        if facts.count is None or index >= facts.count:
            count = "an unknown number of" if facts.count is None else str(facts.count)
            raise DeviceUnavailable(f"index {index} is not among this host's {count} {spec.kind} device(s)")


def _indexed_name(spec: DeviceSpec, facts: HostFacts) -> str:
    """Return the descriptions of the named devices joined by "; ", or raise DeviceUnavailable for one not usable.

    A device the probe refused is refused with the probe's reason; one it
    does not know with the numbers it can use.
    """
    usable = facts.by_index or {}
    refused = facts.refused or {}
    for index in spec.indices:
        if index in usable:
            continue
        if index in refused:
            raise DeviceUnavailable(f"{spec.kind} device {index} cannot be used: {refused[index]}")
        known = ", ".join(str(number) for number in sorted(usable)) or "none"
        raise DeviceUnavailable(f"index {index} is not among this host's usable {spec.kind} devices ({known})")
    return "; ".join(usable[index] for index in spec.indices)


def _check_framework(kind: str, framework: FrameworkBuild | None) -> None:
    """Raise DeviceUnavailable when the framework build cannot reach a device of `kind`."""
    if framework is None or kind == "cpu":
        return
    label = f"{framework.name} {framework.version}"
    if kind == TENSTORRENT and framework.name not in TENSTORRENT_FRAMEWORKS:
        raise DeviceUnavailable(
            f"{label} cannot reach a tenstorrent device; only {', '.join(TENSTORRENT_FRAMEWORKS)} builds can"
        )
    if kind == "rocm" and framework.hip is None:
        raise DeviceUnavailable(f"{label} is not a HIP build, so it cannot reach a rocm device")
    if kind == "cuda" and (framework.cuda is None or framework.hip is not None):
        raise DeviceUnavailable(f"{label} is not a CUDA build, so it cannot reach a cuda device")


def _runtime(kind: str, framework: FrameworkBuild | None, facts: HostFacts) -> str | None:
    """Return the device runtime a record names: None for cpu and tenstorrent, else the build's or the probe's."""
    if kind in ("cpu", TENSTORRENT):
        return None
    if framework is None:
        return facts.runtime
    return framework.cuda if kind == "cuda" else framework.hip


def device_driver(records: Iterable[DeviceRecord]) -> str | None:
    """Return the distinct drivers of `records` in record order, joined by "; ", or None when none names one."""
    drivers: list[str] = []
    for item in records:
        if item.driver is not None and item.driver not in drivers:
            drivers.append(item.driver)
    return "; ".join(drivers) or None
