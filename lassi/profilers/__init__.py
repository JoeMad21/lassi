"""Profilers: the Profiler components (task P17.7; bible Component Interfaces, Profiler row; Result Record, Profile).

Importing this package registers the profilers in
lassi.core.registry.DEFAULT_REGISTRY:

- "timing" (timing.TimingProfiler): the window's runtime only; no
  capability, no config key.
- "nvml" (nvml.NvmlProfiler): an NVIDIA GPU's power through the NVML
  library; declares supports_power and takes_device, takes interval_ms.
- "rocm_smi" (rocm_smi.RocmSmiProfiler): an AMD GPU's power from amdgpu's
  hwmon power1_average in sysfs; declares supports_power and
  takes_device, takes interval_ms.

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

from lassi.profilers import nvml, power, rocm_smi, timing
from lassi.profilers.nvml import NvmlProfiler
from lassi.profilers.power import PowerSource, TelemetryError, TelemetryUnavailable
from lassi.profilers.rocm_smi import RocmSmiProfiler
from lassi.profilers.timing import TimingProfiler

__all__ = [
    "NvmlProfiler",
    "PowerSource",
    "RocmSmiProfiler",
    "TelemetryError",
    "TelemetryUnavailable",
    "TimingProfiler",
    "nvml",
    "power",
    "rocm_smi",
    "timing",
]
