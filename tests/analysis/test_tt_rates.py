"""Tests for the Tenstorrent rates in lassi.analysis (task P17.42; plans/spikes/tt-perf.md, (D) and (F)).

Bible: Evaluation Protocol (Reporting Rules), Result Record (DeviceTiming),
Training Module (Compute), Agent Rules 1, 2, and 14; the owner's answer of
2026-10-26 to OQ-106, option (a): analytic useful work per
item, rates only over a measured silicon time naming the time, a percent of
the vendor's peak shown labelled, PaLM model FLOPs for training, and never a
rate from ttsim.

lassi.analysis.tt_rates:

- check_silicon(device) raises RateRefused for a device that is unnamed or
  names a simulator (ttsim's device line says "a simulator"), and for one
  that does not say "(silicon)": no rate is ever computed from ttsim
  (Agent Rule 2).
- run_rates(run, work, device, name=..., peak=..., wall=False) gives, for one
  run's RunInfo and the item's Work, the FLOP rate and the byte rate over
  the device kernel time summed over its programs (tt-metal's DEVICE
  KERNEL DURATION [ns]); each Rate names its numerator and its
  denominator, the AICLK and the build, and says when the reading is
  self-reported. A FLOP rate carries the percent of `peak` and the peak's
  label (its figure and its source); None means no percent. A rate whose
  work count or time is missing has value None and a note saying why.
  With wall=True it adds the rates over the process wall time, labelled
  as including device open, JIT, transfers, and close.
- trial_rates(trial, work, ...) checks the trial's device once, then gives
  the reference run's rates and each attempt's.
- WORMHOLE_CHIP_PEAK is 65.5e12 FLOP/s, half of the vendor's 131 TFLOP/s
  FP16 per n300 card, its label naming the source and that the counting is
  unstated.
- training_summary(steps, model_flops, device) summarizes a training run's
  steps.jsonl over the steps without a compile: the median step time, the
  tokens and padded tokens per second, and the model FLOP/s, labelled
  "analytic model FLOPs / measured step time", with the steps it left out;
  it refuses a simulator device too.

Every record is SYNTHETIC; no value is a measurement.
"""

from __future__ import annotations

import importlib
import math
from types import ModuleType
from typing import Any

import pytest

from lassi.bench.registry import Work
from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Attempt,
    BenchItem,
    DeviceProgram,
    DeviceTiming,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
)

SILICON = (
    "Tenstorrent Wormhole n300 x 8 (silicon): card 0 serial S0, firmware bundle B, ARC firmware A; "
    "tt-metal 5280a9cf-tracy"
)
TTSIM = (
    "ttsim v1.3.4 (libttsim_wh.so, a virtual Wormhole) on the host CPU, with tt-metal 5280a9cf-tracy: a "
    "simulator, not silicon"
)
MATMUL = Work(flop=524_288_000, ops={"mac": 262_144_000}, bytes=2_457_600, basis="SYNTHETIC: 2MNK")
LOOPBACK = Work(flop=None, ops={}, bytes=204_800, basis="SYNTHETIC: a copy, no floating-point arithmetic")


@pytest.fixture(scope="module")
def module() -> ModuleType:
    """Import lassi.analysis.tt_rates, failing clearly while task P17.42 has not added it."""
    try:
        return importlib.import_module("lassi.analysis.tt_rates")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.analysis.tt_rates does not exist yet (task P17.42): {error}")


def timing(*kernel_ns: int, self_reported: bool = False) -> DeviceTiming:
    """Return a SYNTHETIC reading with one program per kernel time (cycles equal to ns at 1000 MHz)."""
    programs = [DeviceProgram(device_id=0, op_name="", call_count=1024 * (i + 1), cores=64, kernel_cycles=ns,
                              kernel_ns=ns, fw_ns=None) for i, ns in enumerate(kernel_ns)]
    return DeviceTiming(source="tt-metal device profiler", build="tt-metal@5280a9cf-tracy", arch="wormhole_b0",
                        aiclk_mhz=1000, programs=programs, self_reported=self_reported)


def trial(device: str | None, reference: RunInfo, *attempts: RunInfo) -> Trial:
    """Return a SYNTHETIC trial with the reference run and one attempt per run."""
    provenance = Provenance(commit=None, dirty=None, device=device, sdk=None, date="2026-10-26T12:00:00+00:00")
    return Trial(
        trial_id="lassi-demo/arm/tt-pairs-v0/cpp-tt/matmul_multi_core/run01",
        recipe_hash="0" * 64,
        provenance=provenance,
        bench_item=BenchItem(suite="tt-pairs-v0", item="matmul_multi_core", split="unassigned", direction="cpp-tt"),
        model=ModelInfo(backend="none", id="none", sampling=Sampling(temperature=0.0, top_p=1.0, max_tokens=1)),
        reference_run=reference,
        attempts=[Attempt(index=i, stage_reached="S5", run=run) for i, run in enumerate(attempts)],
    )


def by_kind(rates: list[Any], run: str) -> dict[str, Any]:
    """Return the rates of one run keyed by kind."""
    return {rate.kind: rate for rate in rates if rate.run == run}


def test_the_peak_is_half_the_cards_and_names_its_source(module: ModuleType) -> None:
    peak = module.WORMHOLE_CHIP_PEAK
    assert peak.flop_per_s == 65.5e12
    label = peak.label()
    for words in ("65.5 TFLOP/s", "131", "n300", "unstated", "FP16"):
        assert words in label, (words, label)


@pytest.mark.parametrize("device", [None, "", TTSIM, "ttsim v1.3.4 on tt-metal 5280a9cf", "Intel Xeon (host CPU)"])
def test_no_rate_comes_from_a_simulator_or_an_unnamed_device(module: ModuleType, device: str | None) -> None:
    with pytest.raises(module.RateRefused):
        module.check_silicon(device)
    if device == "":
        return  # a Provenance never holds an empty device
    run = RunInfo(exit_code=0, hang=False, wall_s=2.0, device_timing=timing(8000))
    with pytest.raises(module.RateRefused, match="Agent Rule 2|silicon"):
        module.trial_rates(trial(device, run), MATMUL)


def test_a_matmul_on_silicon_gets_its_flop_and_byte_rates_over_kernel_time(module: ModuleType) -> None:
    run = RunInfo(exit_code=0, hang=False, wall_s=1.8, device_timing=timing(6000, 2000))
    rates = by_kind(module.run_rates(run, MATMUL, SILICON, name="reference"), "reference")
    flop, data = rates["flop_per_s"], rates["bytes_per_s"]
    assert math.isclose(flop.value, 524_288_000 / 8000e-9)
    assert math.isclose(data.value, 2_457_600 / 8000e-9)
    assert (flop.work, flop.seconds) == (524_288_000, 8000e-9)
    assert math.isclose(flop.percent_of_peak, 524_288_000 / 8000e-9 / 65.5e12 * 100)
    assert flop.peak_label == module.WORMHOLE_CHIP_PEAK.label()
    assert data.percent_of_peak is None and data.peak_label is None
    for words in ("analytic", "device kernel time", "2 program", "8000 ns", "AICLK 1000 MHz",
                  "tt-metal device profiler", "tt-metal@5280a9cf-tracy"):
        assert words in flop.label, (words, flop.label)
    assert "wall" not in flop.label


def test_no_peak_means_no_percent(module: ModuleType) -> None:
    run = RunInfo(exit_code=0, hang=False, device_timing=timing(8000))
    (flop,) = [rate for rate in module.run_rates(run, MATMUL, SILICON, name="attempt 0", peak=None)
               if rate.kind == "flop_per_s"]
    assert flop.value is not None and flop.percent_of_peak is None and flop.peak_label is None


def test_an_item_without_a_flop_count_gets_its_byte_rate_only(module: ModuleType) -> None:
    run = RunInfo(exit_code=0, hang=False, device_timing=timing(4000))
    rates = by_kind(module.run_rates(run, LOOPBACK, SILICON, name="reference"), "reference")
    assert rates["flop_per_s"].value is None and "no FLOP count" in rates["flop_per_s"].note
    assert math.isclose(rates["bytes_per_s"].value, 204_800 / 4000e-9)


def test_a_run_without_a_reading_gets_no_rate_and_says_why(module: ModuleType) -> None:
    run = RunInfo(exit_code=0, hang=False, wall_s=1.8)
    rates = module.run_rates(run, MATMUL, SILICON, name="reference")
    assert rates and all(rate.value is None for rate in rates)
    assert all("no device time" in rate.note for rate in rates)


def test_a_self_reported_reading_is_labelled_so(module: ModuleType) -> None:
    run = RunInfo(exit_code=0, hang=False, device_timing=timing(8000, self_reported=True))
    (flop,) = [rate for rate in module.run_rates(run, MATMUL, SILICON, name="attempt 0") if rate.kind == "flop_per_s"]
    assert flop.value is not None and "self-reported" in flop.note


def test_rates_over_wall_time_only_on_request_and_labelled(module: ModuleType) -> None:
    run = RunInfo(exit_code=0, hang=False, wall_s=2.0, device_timing=timing(8000))
    assert {rate.kind for rate in module.run_rates(run, MATMUL, SILICON, name="r")} == {"flop_per_s", "bytes_per_s"}
    rates = by_kind(module.run_rates(run, MATMUL, SILICON, name="r", wall=True), "r")
    wall = rates["flop_per_s_wall"]
    assert math.isclose(wall.value, 524_288_000 / 2.0) and wall.percent_of_peak is None
    for words in ("process wall time", "device open", "JIT", "transfers", "close"):
        assert words in wall.label, (words, wall.label)


def test_trial_rates_give_the_reference_then_each_attempt(module: ModuleType) -> None:
    reference = RunInfo(exit_code=0, hang=False, device_timing=timing(8000))
    attempt = RunInfo(exit_code=0, hang=False, device_timing=timing(16000, self_reported=True))
    rates = module.trial_rates(trial(SILICON, reference, attempt), MATMUL)
    assert [rate.run for rate in rates] == ["reference", "reference", "attempt 0", "attempt 0"]
    assert math.isclose(by_kind(rates, "attempt 0")["flop_per_s"].value, 524_288_000 / 16000e-9)


# ---------------------------------------------------------------------------
# Training


def step(number: int, seconds: float | None, compile_step: bool, tokens: int = 40) -> dict[str, Any]:
    """Return a SYNTHETIC steps.jsonl line of the ttxla worker."""
    return {"step": number, "loss": 1.0, "step_time_s": seconds, "tokens": tokens, "tokens_computed": 64,
            "model_flop": 64_000, "compiles": 1 if compile_step else 0, "compile_step": compile_step}


FLOPS = {"convention": "palm-model-flops", "flop_per_token": 1000}


def test_the_training_summary_leaves_out_compile_steps(module: ModuleType) -> None:
    steps = [step(1, 10.0, True), step(2, 0.5, False), step(3, 0.5, False), step(4, 1.0, False)]
    summary = module.training_summary(steps, FLOPS, "tenstorrent card 0 (xla:0)")
    assert summary["steps_counted"] == 3 and summary["steps_left_out"] == [1]
    assert math.isclose(summary["median_step_time_s"], 0.5)
    assert math.isclose(summary["tokens_per_s"], 120 / 2.0)
    assert math.isclose(summary["tokens_computed_per_s"], 192 / 2.0)
    assert math.isclose(summary["model_flop_per_s"], 192_000 / 2.0)
    assert summary["convention"] == "palm-model-flops" and summary["device"] == "tenstorrent card 0 (xla:0)"
    assert "analytic model FLOPs / measured step time" in summary["labels"]["model_flop_per_s"]


def test_a_summary_with_no_step_left_has_no_rates_and_says_why(module: ModuleType) -> None:
    summary = module.training_summary([step(1, 10.0, True), step(2, None, False)], FLOPS, "tenstorrent card 0")
    assert summary["steps_counted"] == 0 and summary["steps_left_out"] == [1, 2]
    assert summary["median_step_time_s"] is None and summary["model_flop_per_s"] is None
    assert summary["note"]


def test_a_summary_without_a_flop_convention_has_no_flop_rate(module: ModuleType) -> None:
    summary = module.training_summary([step(1, 0.5, False)], None, "tenstorrent card 0")
    assert summary["model_flop_per_s"] is None and summary["tokens_per_s"] is not None


def test_the_training_summary_refuses_a_simulator(module: ModuleType) -> None:
    with pytest.raises(module.RateRefused):
        module.training_summary([step(1, 0.5, False)], FLOPS, TTSIM)
