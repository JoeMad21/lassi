"""Profilers: the Profiler components (tasks P17.7, P9.1; bible Component Interfaces, Profiler row; Result Record).

Importing this package registers the profilers in
lassi.core.registry.DEFAULT_REGISTRY:

- "timing" (timing.TimingProfiler): the window's runtime only; no
  capability, no config key.
- "nvml" (nvml.NvmlProfiler): an NVIDIA GPU's power through the NVML
  library; declares supports_power and takes_device, takes interval_ms.
- "rocm_smi" (rocm_smi.RocmSmiProfiler): an AMD GPU's power from amdgpu's
  hwmon power1_average in sysfs; declares supports_power and
  takes_device, takes interval_ms.
- "energy" (energy.EnergyProfiler, task P9.1): energy, power, or runtime
  of one device through the backend profiler.backend names, with no
  default; declares supports_power and takes_device, takes backend and
  each backend's keys (interval_ms, package). Its backends register in
  energy.BACKENDS: wall (runtime only, labeled "wall time, not energy") and
  rapl (rapl.RaplSource, a CPU package's energy counter in powercap
  sysfs). energy.TelemetrySource is the seam, which takes an energy
  counter or power readings, and a backend that cannot measure refuses
  when built.

The power profilers read telemetry through a source seam
(power.PowerSource) and share the sampler and the trapezoid-rule integral
(power.Sampler, power.summarize). Device files and telemetry are read only
here and in lassi/executors (Design Principle 9). No vendor library is
imported when this package is imported: pynvml is imported only when an
nvml source is built (the registration rule). No profiler starts a process
or opens a device node, except NVML's own initialization, which runs only
when an nvml profiler is built, after the cuda probe of its device passed.
"""

from __future__ import annotations

from lassi.profilers import energy, nvml, power, rapl, rocm_smi, timing
from lassi.profilers.energy import EnergyProfiler, TelemetrySource
from lassi.profilers.nvml import NvmlProfiler
from lassi.profilers.power import PowerSource, TelemetryError, TelemetryUnavailable
from lassi.profilers.rocm_smi import RocmSmiProfiler
from lassi.profilers.timing import TimingProfiler

__all__ = [
    "EnergyProfiler",
    "NvmlProfiler",
    "PowerSource",
    "RocmSmiProfiler",
    "TelemetryError",
    "TelemetrySource",
    "TelemetryUnavailable",
    "TimingProfiler",
    "energy",
    "nvml",
    "power",
    "rapl",
    "rocm_smi",
    "timing",
]
