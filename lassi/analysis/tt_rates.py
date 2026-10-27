"""Tenstorrent rates: analytic work over measured silicon time, labelled (task P17.42; tt-perf spike, (D), (F)).

Bible: Evaluation Protocol (Reporting Rules), Result Record (DeviceTiming),
Training Module (Compute), Agent Rules 1, 2, and 14. The owner's answer of
2026-10-26 to OQ-106, option (a): FLOPs are the analytic
useful work of an item (lassi.bench.registry Work) or of a training step
(PaLM model FLOPs, lassi.train.ttxla_worker), the same for every candidate
and never counted on the device; a rate divides that work by a measured
silicon time and names the time; a percent of the vendor's peak is shown
labelled with the peak's figure and source; and no rate ever comes from
ttsim, whose clock follows the host's reads (Agent Rule 2).

Rates are derived here, never stored as measurements: the record keeps
the raw parts (RunInfo.device_timing's cycles, nanoseconds, AICLK, and
build; the manifest's work; steps.jsonl's step times and counts), and every
Rate carries them in its label beside the value (Agent Rule 1).

- check_silicon(device) refuses (RateRefused) a device that is unnamed,
  names a simulator or ttsim, or does not say "(silicon)", the tt_silicon
  executor's device line.
- run_rates(run, work, device, name=..., peak=WORMHOLE_CHIP_PEAK,
  wall=False): the FLOP rate and the byte rate of one run over its device
  kernel time, the DEVICE KERNEL DURATION [ns] of its programs summed; the
  FLOP rate with its percent of `peak` (None: no percent). A missing work
  count or reading gives a Rate with value None and a note saying why; a
  self-reported reading (a model-written program's) is computed and
  labelled so. With wall=True, the two rates over the run's process wall
  time follow, labelled as including device open, JIT, transfers, and close.
- trial_rates(trial, work, ...) checks the trial's device, then gives the
  reference run's rates and each attempt's, in order.
- training_summary(steps, model_flops, device) summarizes a training run's
  steps over those without a graph compile, refusing a simulator device.

WORMHOLE_CHIP_PEAK is half of the vendor's 131 TFLOP/s FP16 per n300 card
(docs.tenstorrent.com, the n300 specification, read 2026-10-26), since a
run's programs execute on one of the card's two chips; the vendor states no
counting method, and the items run bf16, so every percent says so.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from lassi.bench.registry import Work
from lassi.core.record import DeviceTiming, RunInfo, Trial

# A device line that names a simulator: the ttsim executor's says "a simulator, not silicon" (Agent Rule 2).
SIMULATOR_WORDS = ("simulator", "ttsim")
# The tt_silicon executor's device line holds this (lassi.executors.tt_silicon device()).
SILICON_MARK = "(silicon)"
# The training summary's label of its model FLOP rate (the spike, (F)).
TRAINING_FLOP_LABEL = "analytic model FLOPs / measured step time"


class RateRefused(ValueError):
    """A rate the Reporting Rules forbid: from a simulator run, or from a device not named as silicon."""


@dataclass(frozen=True)
class Peak:
    """A vendor's peak rate, the scope it applies to, and where it was read."""

    flop_per_s: float
    scope: str
    source: str

    def label(self) -> str:
        """Return the peak as a report shows it: its figure, its scope, and its source."""
        return f"{self.flop_per_s / 1e12:g} TFLOP/s {self.scope} ({self.source})"


WORMHOLE_CHIP_PEAK = Peak(
    flop_per_s=65.5e12,
    scope="FP16 per Wormhole chip, half of the vendor's 131 TFLOP/s per n300 card",
    source="the vendor's n300 specification, docs.tenstorrent.com, read 2026-10-26; its counting is unstated, and the "
           "items run bf16",
)


@dataclass(frozen=True)
class Rate:
    """One derived rate of one run: its value (None when it cannot be computed), its parts, and its labels.

    `run` names the run ("reference", "attempt <n>"), `kind` the rate
    (flop_per_s, bytes_per_s, and their _wall forms), `work` the analytic
    numerator, `seconds` the measured denominator, `label` both and their
    source, `percent_of_peak` and `peak_label` the percent of the vendor's
    peak (FLOP rates over device time only), and `note` why a value is
    None or that the reading is self-reported.
    """

    run: str
    kind: str
    value: float | None
    work: int | None
    seconds: float | None
    label: str
    percent_of_peak: float | None = None
    peak_label: str | None = None
    note: str | None = None


def refuse_simulator(device: str | None) -> None:
    """Raise RateRefused when `device` names a simulator (Agent Rule 2: ttsim results never carry performance)."""
    lowered = (device or "").lower()
    if any(word in lowered for word in SIMULATOR_WORDS):
        raise RateRefused(f"no rate comes from a simulator run (Agent Rule 2); the device is {device!r}")


def check_silicon(device: str | None) -> None:
    """Raise RateRefused unless `device` is named, names no simulator, and says it is silicon."""
    if not device:
        raise RateRefused("the run names no device, so no rate can say it was measured on silicon (Agent Rule 1)")
    refuse_simulator(device)
    if SILICON_MARK not in device:
        raise RateRefused(f"rates come only from runs on the Tenstorrent cards; the device {device!r} is not silicon")


def _kernel_label(timing: DeviceTiming, work_text: str) -> tuple[float, str]:
    """Return (seconds, label) of a reading's device kernel time summed over its programs."""
    total_ns = sum(program.kernel_ns for program in timing.programs)
    cycles = sum(program.kernel_cycles for program in timing.programs)
    count = len(timing.programs)
    label = (
        f"analytic {work_text} / device kernel time, {count} program(s), {total_ns} ns ({cycles} cycles at AICLK "
        f"{timing.aiclk_mhz} MHz; {timing.source}, {timing.build})"
    )
    return total_ns / 1e9, label


def _device_rates(run: RunInfo, work: Work, name: str, peak: Peak | None) -> list[Rate]:
    """Return the FLOP and byte rates of one run over its device kernel time (see the module docstring)."""
    timing = run.device_timing
    if timing is None:
        note = "no device time: the run was not profiled (device_profiler off), or its reading had a shortfall"
        return [Rate(run=name, kind=kind, value=None, work=count, seconds=None,
                     label=f"analytic {count} {unit} / device kernel time (not measured)", note=note)
                for kind, count, unit in (("flop_per_s", work.flop, "FLOP"), ("bytes_per_s", work.bytes, "bytes"))]
    note = "self-reported: the run's own process wrote the profiler's files (model-written code)" \
        if timing.self_reported else None
    rates = []
    seconds, label = _kernel_label(timing, f"{work.flop} FLOP" if work.flop is not None else "FLOP")
    if work.flop is None or seconds <= 0:
        why = f"the item declares no FLOP count ({work.basis})" if work.flop is None else "a kernel time of 0 ns"
        rates.append(Rate(run=name, kind="flop_per_s", value=None, work=work.flop, seconds=seconds, label=label,
                          note=why))
    else:
        value = work.flop / seconds
        rates.append(Rate(run=name, kind="flop_per_s", value=value, work=work.flop, seconds=seconds, label=label,
                          percent_of_peak=None if peak is None else value / peak.flop_per_s * 100,
                          peak_label=None if peak is None else peak.label(), note=note))
    seconds, label = _kernel_label(timing, f"{work.bytes} bytes moved")
    value = work.bytes / seconds if seconds > 0 else None
    rates.append(Rate(run=name, kind="bytes_per_s", value=value, work=work.bytes, seconds=seconds, label=label,
                      note=note if value is not None else "a kernel time of 0 ns"))
    return rates


def _wall_rates(run: RunInfo, work: Work, name: str) -> list[Rate]:
    """Return the FLOP and byte rates of one run over its process wall time, labelled as what that time holds."""
    seconds = run.wall_s
    text = "over process wall time (device open, JIT, transfers, and close included; the sandbox's whole command)"
    rates = []
    for kind, count, unit in (("flop_per_s_wall", work.flop, "FLOP"), ("bytes_per_s_wall", work.bytes, "bytes")):
        label = f"analytic {count} {unit} {text}, {seconds} s"
        if count is None or not seconds:
            note = "no work count or no wall time"
            rates.append(Rate(run=name, kind=kind, value=None, work=count, seconds=seconds, label=label, note=note))
        else:
            rates.append(Rate(run=name, kind=kind, value=count / seconds, work=count, seconds=seconds, label=label))
    return rates


def run_rates(
    run: RunInfo, work: Work, device: str | None, *, name: str, peak: Peak | None = WORMHOLE_CHIP_PEAK,
    wall: bool = False,
) -> list[Rate]:
    """Return one run's rates: over its device kernel time, then, with `wall`, over its process wall time.

    RateRefused when `device` is not silicon (check_silicon).
    """
    check_silicon(device)
    rates = _device_rates(run, work, name, peak)
    return [*rates, *_wall_rates(run, work, name)] if wall else rates


def trial_rates(
    trial: Trial, work: Work, *, peak: Peak | None = WORMHOLE_CHIP_PEAK, wall: bool = False
) -> list[Rate]:
    """Return the reference run's rates, then each attempt's, after checking the trial's device once."""
    device = trial.provenance.device
    check_silicon(device)
    rates = run_rates(trial.reference_run, work, device, name="reference", peak=peak, wall=wall)
    for attempt in trial.attempts:
        rates += run_rates(attempt.run, work, device, name=f"attempt {attempt.index}", peak=peak, wall=wall)
    return rates


def _counted(step: Mapping[str, Any]) -> bool:
    """Return whether a training step counts: no compile, and a step time above 0."""
    seconds = step.get("step_time_s")
    return step.get("compile_step") is False and isinstance(seconds, (int, float)) and seconds > 0


def training_summary(
    steps: Sequence[Mapping[str, Any]], model_flops: Mapping[str, Any] | None, device: str | None
) -> dict[str, Any]:
    """Return a training run's performance summary over its steps without a compile (the spike, (F)).

    The median step time, and the tokens, the padded tokens, and the model
    FLOPs per second as sums over sums, each labelled; the steps left out
    are named. Values are None, with a note, when no step counts or a part
    is missing. RateRefused for a simulator device.
    """
    refuse_simulator(device)
    counted = [step for step in steps if _counted(step)]
    left_out = [step.get("step") for step in steps if not _counted(step)]
    seconds = sum(float(step["step_time_s"]) for step in counted)
    flops_known = model_flops is not None and all(isinstance(step.get("model_flop"), int) for step in counted)

    def per_second(key: str) -> float | None:
        values = [step.get(key) for step in counted]
        if not counted or not all(isinstance(value, int) for value in values):
            return None
        return sum(values) / seconds

    return {
        "convention": None if model_flops is None else model_flops.get("convention"),
        "device": device,
        "steps_counted": len(counted),
        "steps_left_out": left_out,
        "median_step_time_s": statistics.median(float(step["step_time_s"]) for step in counted) if counted else None,
        "tokens_per_s": per_second("tokens"),
        "tokens_computed_per_s": per_second("tokens_computed"),
        "model_flop_per_s": per_second("model_flop") if flops_known else None,
        "labels": {
            "median_step_time_s": "host monotonic time per step, from its start to after its device sync",
            "tokens_per_s": "non-pad tokens / measured step time",
            "tokens_computed_per_s": "padded tokens run / measured step time",
            "model_flop_per_s": TRAINING_FLOP_LABEL,
        },
        "note": None if counted else "no step without a compile and with a step time; nothing is summarized",
    }
