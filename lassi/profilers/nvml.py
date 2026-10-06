"""The nvml profiler, registered as Profiler "nvml" (task P17.7): NVIDIA GPU power through the NVML library.

Its recipe section names one cuda GPU and the sampling interval:
`profiler: {kind: nvml, device: {kind: cuda, indices: [0]}, interval_ms: 10}`.
It declares supports_power and takes_device; sampling, the integral, and the
settings are lassi.profilers.power's.

Its real source, NvmlSource, imports pynvml (distribution nvidia-ml-py)
when it is built, never when this module is imported (the registration
rule); without it the build is TelemetryUnavailable. No dependency is
declared for it yet. It maps the recipe's index as the gpu executor does:
index i is the i-th entry of <root>/proc/driver/nvidia/gpus in name order,
each entry named by its PCI bus id (NVIDIA/open-gpu-kernel-modules
615.71.09, kernel-open/nvidia/nv-procfs.c:1451-1453), which is the order
lassi.executors.devices gpu_inventory uses. It takes the handle by that bus
id, never by NVML's own index, whose order is not the PCI order.

The calls, read in nvidia-ml-py 13.615.71 (pynvml.py): nvmlInit() calls
nvmlInitWithFlags(0) (:3161-3163); nvmlDeviceGetHandleByPciBusId(bus id)
encodes a str argument to bytes (convertStrBytes, :3116) and calls
nvmlDeviceGetHandleByPciBusId_v2 (:3421-3428); nvmlDeviceGetPowerUsage(handle)
returns the C unsigned int as it comes (:3962-3967); a failed call raises
NVMLError (_nvmlCheckReturn, :1114-1116). That the power is in milliwatts,
and the bus id form NVML accepts, come from NVML's documentation, not from
code (NVML is closed source): unverified until P17.13. nvmlInit opens the
NVIDIA device nodes, so it runs only when the profiler is built, after the
cuda probe of profiler.device passed; nvmlInit is kept for the run, and
process exit ends it.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

from lassi.core.registry import register
from lassi.profilers.power import PowerProfiler, PowerSource, TelemetryError, TelemetryUnavailable

# The NVIDIA procfs directory with one entry per GPU, named by its PCI bus id, under the host root.
_NVIDIA_GPUS = ("proc", "driver", "nvidia", "gpus")
# NVML reports power in milliwatts (NVML documentation; unverified until P17.13).
_MILLIWATTS_PER_WATT = 1000


def _bus_id(index: int, root: Path) -> str:
    """Return the PCI bus id of cuda GPU `index`: the index-th entry name of proc/driver/nvidia/gpus, in name order.

    TelemetryUnavailable when the host lists no such GPU; nothing is opened.
    """
    directory = root.joinpath(*_NVIDIA_GPUS)
    try:
        with os.scandir(directory) as found:
            names = sorted(entry.name for entry in found)
    except OSError:
        names = []
    if index >= len(names):
        raise TelemetryUnavailable(
            f"cuda index {index}: /proc/driver/nvidia/gpus lists {len(names)} GPUs, so there is no power to read"
        )
    return names[index]


def _library() -> ModuleType:
    """Import pynvml now; TelemetryUnavailable naming the distribution when it is not installed."""
    try:
        return importlib.import_module("pynvml")
    except ImportError as error:
        raise TelemetryUnavailable(
            f"the nvml profiler reads power with the module pynvml (distribution nvidia-ml-py), which cannot be "
            f"imported: {error}"
        ) from None


class NvmlSource:
    """The power of one NVIDIA GPU through NVML, its handle taken by the PCI bus id of the recipe's index."""

    def __init__(self, *, index: int, root: Path) -> None:
        """Find the bus id, import pynvml, initialize NVML, and take the handle; TelemetryUnavailable on failure."""
        bus = _bus_id(index, Path(root))
        nvml = _library()
        try:
            nvml.nvmlInit()
            handle: Any = nvml.nvmlDeviceGetHandleByPciBusId(bus)
        except nvml.NVMLError as error:
            raise TelemetryUnavailable(f"cuda index {index} (PCI bus id {bus}): NVML refused: {error}") from None
        self._nvml = nvml
        self._handle = handle
        self._where = f"cuda index {index} (PCI bus id {bus})"

    def read_w(self) -> float:
        """Return the GPU's power usage in watts (NVML's milliwatts / 1000); TelemetryError when NVML fails."""
        try:
            milliwatts = self._nvml.nvmlDeviceGetPowerUsage(self._handle)
        except self._nvml.NVMLError as error:
            raise TelemetryError(f"{self._where}: nvmlDeviceGetPowerUsage failed: {error}") from None
        return milliwatts / _MILLIWATTS_PER_WATT


@register("Profiler", "nvml")
class NvmlProfiler(PowerProfiler):
    """Power of one cuda GPU through NVML, sampled at profiler.interval_ms (lassi.profilers.power)."""

    name: ClassVar[str] = "nvml"
    KIND: ClassVar[str] = "cuda"

    def _open_source(self, index: int, root: Path) -> PowerSource:
        """Return the NvmlSource of cuda GPU `index` under `root`."""
        return NvmlSource(index=index, root=root)
