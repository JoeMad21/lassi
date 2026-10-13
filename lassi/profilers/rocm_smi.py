"""The rocm_smi profiler, registered as Profiler "rocm_smi" (task P17.7): AMD GPU power from amdgpu's hwmon file.

Its recipe section names one rocm GPU and the sampling interval:
`profiler: {kind: rocm_smi, device: {kind: rocm, indices: [0]}, interval_ms: 10}`.
It declares supports_power and takes_device; sampling, the integral, and the
settings are lassi.profilers.power's.

Its real source, HwmonSource, starts no tool and opens no device node
(OQ-002): it reads the sysfs attribute that the ROCm SMI library reads for
average power, with file metadata, directory listings, and sysfs text only,
as lassi.executors.devices reads. Index i maps as the gpu executor maps it:
lassi.executors.devices gpu_inventory("rocm", root)[i] names the render node
/dev/dri/renderD<m> (KFD topology order), of which only the name renderD<m>
is used; the node itself is never touched. The power file is
<root>/sys/class/drm/renderD<m>/device/hwmon/<the one hwmon directory>/
power1_average, read anew at each read, in microwatts.

Sources. ROCm/rocm-systems rocm-7.2.1, projects/rocm-smi-lib/src:
rsmi_dev_power_ave_get reads monitor kMonPowerAve with sensor index 1
(rocm_smi.cc:3197, :3203), whose file name is "power#_average" with '#' the
sensor index (rocm_smi_monitor.cc:80, :138, :299), in the hwmon directory
the library finds under /sys/class/drm/<device>/device/hwmon
(rocm_smi_main.cc:72, :676; it takes the first entry whose name file holds
an AMD monitor type, :617-647, where this source requires exactly one
entry). Linux v6.12, drivers/gpu/drm/amd/pm/amdgpu_pm.c: power1_average
shows the GPU's average power sensor (:3209-3220, :3522) in microwatts
(:3204), and is hidden on a GPU without that sensor (:3679-3681). Not read
from source and unverified until P17.12 and P17.13 measure the GPU host:
that renderD<m>/device leads to the same device as the library's card
entry, which attribute the GPU host's amdgpu exposes (power1_average, or
only power1_input, which the library's rsmi_dev_power_get tries first,
rocm_smi.cc:3309-3314), and how often it updates.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import ClassVar

from lassi.core.registry import register
from lassi.executors.devices import gpu_inventory
from lassi.profilers.power import PowerProfiler, PowerSource, TelemetryError, TelemetryUnavailable

# The DRM class directory under the host root, holding one entry per render node.
_DRM = PurePosixPath("/sys/class/drm")
# The average power attribute of an amdgpu hwmon directory, in microwatts (amdgpu_pm.c:3204, :3522).
_POWER_FILE = "power1_average"
_MICROWATTS_PER_WATT = 1_000_000


def _under(root: Path, host: PurePosixPath) -> Path:
    """Return the host path `host` (absolute, POSIX) under the root `root`."""
    return root.joinpath(*host.parts[1:])


def _hwmon_power_file(index: int, root: Path) -> PurePosixPath:
    """Return the host path of rocm GPU `index`'s power1_average file; TelemetryUnavailable when it has none.

    The index maps through the KFD inventory to its render node's name;
    that node's device must hold exactly one hwmon directory. Messages name
    host paths, never the root they were read under.
    """
    inventory = gpu_inventory("rocm", root)
    if index >= len(inventory):
        raise TelemetryUnavailable(
            f"rocm index {index}: the KFD topology lists {len(inventory)} GPUs, so there is no power to read"
        )
    node = inventory[index].node
    if node is None:
        raise TelemetryUnavailable(f"rocm index {index}: its KFD node names no render node (drm_render_minor)")
    hwmon = _DRM / PurePosixPath(node).name / "device" / "hwmon"
    try:
        with os.scandir(_under(root, hwmon)) as found:
            names = sorted(entry.name for entry in found if entry.name.startswith("hwmon"))
    except OSError:
        names = []
    if len(names) != 1:
        listed = ", ".join(names) if names else "none"
        raise TelemetryUnavailable(
            f"rocm index {index}: {hwmon} must hold exactly one hwmon directory to read power from, found {listed}"
        )
    return hwmon / names[0] / _POWER_FILE


class HwmonSource:
    """The power of one AMD GPU from its amdgpu hwmon power1_average file, read anew at each read."""

    def __init__(self, *, index: int, root: Path) -> None:
        """Find the power file of rocm GPU `index` under `root` (TelemetryUnavailable when there is none)."""
        self._host = _hwmon_power_file(index, Path(root))
        self._path = _under(Path(root), self._host)

    def read_w(self) -> float:
        """Return the power in watts (microwatts / 1e6); TelemetryError for a missing, unreadable, or bad value."""
        try:
            text = self._path.read_bytes().decode("ascii").strip()
        except (OSError, UnicodeDecodeError) as error:
            raise TelemetryError(f"{self._host} could not be read: {type(error).__name__}") from None
        if not text.isdigit():
            raise TelemetryError(f"{self._host} holds {text[:40]!r}, not a whole number of microwatts")
        return int(text) / _MICROWATTS_PER_WATT


@register("Profiler", "rocm_smi")
class RocmSmiProfiler(PowerProfiler):
    """Power of one rocm GPU from amdgpu's hwmon power1_average, sampled at profiler.interval_ms."""

    name: ClassVar[str] = "rocm_smi"
    KIND: ClassVar[str] = "rocm"

    def _open_source(self, index: int, root: Path) -> PowerSource:
        """Return the HwmonSource of rocm GPU `index` under `root`."""
        return HwmonSource(index=index, root=root)
