"""Tests for RunInfo.device_timing, the device time of a silicon run (task P17.42).

Bible: Result Record (the run and reference_run blocks; DeviceTiming and
DeviceProgram), Evaluation Protocol (Reporting Rules), Agent Rules 1 and 2;
plans/spikes/tt-perf.md, design (A).

The record these tests fix (lassi.core.record):

- DeviceProgram: device_id, op_name ("" allowed: tt-metal leaves a program
  unnamed), call_count, cores, kernel_cycles, kernel_ns, and fw_ns (None
  when the report left it empty). Counts are integers of at least 0;
  cores and kernel_cycles at least 1.
- DeviceTiming: source and build (non-empty), arch (non-empty), aiclk_mhz
  (at least 1: an AICLK of 0 is what UMD reports for a simulated chip),
  programs (at least one), and self_reported. Both are frozen and
  keyword-only, as every record class.
- RunInfo.device_timing, last, None by default (not measured: every ttsim
  run, every silicon run with device_profiler off, and every trial.json
  written before the field), round-trips through JSON as a nested record.
- The stages copy the executor's reading into the run record, with
  self_reported False for the baseline's runs of the tracked references and
  True for an attempt's run of model-written code.

Every value is SYNTHETIC; none is a measurement.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from lassi.core import record
from lassi.core.record import DeviceProgram, DeviceTiming, RunInfo


def program(**changes: Any) -> DeviceProgram:
    """Return a SYNTHETIC DeviceProgram with `changes`."""
    values: dict[str, Any] = {"device_id": 0, "op_name": "", "call_count": 1024, "cores": 1, "kernel_cycles": 7000,
                              "kernel_ns": 7000, "fw_ns": 9100}
    values.update(changes)
    return DeviceProgram(**values)


def timing(**changes: Any) -> DeviceTiming:
    """Return a SYNTHETIC DeviceTiming with one program and `changes`."""
    values: dict[str, Any] = {"source": "tt-metal device profiler", "build": "tt-metal@5280a9cf-tracy",
                              "arch": "wormhole_b0", "aiclk_mhz": 1000, "programs": [program()],
                              "self_reported": True}
    values.update(changes)
    return DeviceTiming(**values)


def test_the_fields_are_in_order_and_the_run_field_is_last_and_optional() -> None:
    assert [field.name for field in dataclasses.fields(DeviceProgram)] == [
        "device_id", "op_name", "call_count", "cores", "kernel_cycles", "kernel_ns", "fw_ns",
    ]
    assert [field.name for field in dataclasses.fields(DeviceTiming)] == [
        "source", "build", "arch", "aiclk_mhz", "programs", "self_reported",
    ]
    names = [field.name for field in dataclasses.fields(RunInfo)]
    assert names[-1] == "device_timing" and RunInfo().device_timing is None


@pytest.mark.parametrize("cls", [DeviceProgram, DeviceTiming])
def test_the_records_are_frozen_and_keyword_only(cls: type) -> None:
    assert cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert all(field.kw_only for field in dataclasses.fields(cls))


@pytest.mark.parametrize(
    ("changes", "name"),
    [
        ({"device_id": -1}, "device_id"),
        ({"cores": 0}, "cores"),
        ({"kernel_cycles": 0}, "kernel_cycles"),
        ({"kernel_ns": -5}, "kernel_ns"),
        ({"fw_ns": -1}, "fw_ns"),
        ({"call_count": 1.5}, "call_count"),
        ({"op_name": None}, "op_name"),
    ],
)
def test_a_program_checks_its_fields(changes: dict[str, Any], name: str) -> None:
    with pytest.raises(ValueError, match=f"DeviceProgram.{name}"):
        program(**changes)


@pytest.mark.parametrize(
    ("changes", "name"),
    [
        ({"aiclk_mhz": 0}, "aiclk_mhz"),
        ({"programs": []}, "programs"),
        ({"source": ""}, "source"),
        ({"build": ""}, "build"),
        ({"arch": ""}, "arch"),
        ({"self_reported": None}, "self_reported"),
    ],
)
def test_a_timing_checks_its_fields(changes: dict[str, Any], name: str) -> None:
    with pytest.raises(ValueError, match=f"DeviceTiming.{name}"):
        timing(**changes)


def test_a_run_with_a_timing_round_trips_through_json() -> None:
    run = RunInfo(exit_code=0, hang=False, wall_s=1.5, device_timing=timing(programs=[program(), program(fw_ns=None)]))
    data = record.to_dict(run)
    assert list(data)[-1] == "device_timing"
    assert list(data["device_timing"]) == ["source", "build", "arch", "aiclk_mhz", "programs", "self_reported"]
    back = record.from_json(RunInfo, record.to_json(run))
    assert back == run and isinstance(back.device_timing, DeviceTiming)
    assert all(isinstance(item, DeviceProgram) for item in back.device_timing.programs)


def test_an_older_run_record_loads_with_no_timing() -> None:
    data = record.to_dict(RunInfo(exit_code=0, hang=False))
    del data["device_timing"]
    assert record.from_dict(RunInfo, data).device_timing is None


def test_from_dict_refuses_a_broken_timing_naming_it() -> None:
    data = record.to_dict(RunInfo(device_timing=timing()))
    data["device_timing"]["programs"][0]["kernel_cycles"] = 0
    with pytest.raises(ValueError, match="device_timing"):
        record.from_dict(RunInfo, data)


def test_the_stages_keep_the_reading_marking_a_references_run_not_self_reported(tmp_path: Any) -> None:
    import types

    from lassi.core import stages, store
    from lassi.core.interfaces import RunResult

    context = types.SimpleNamespace(store=store.TextStore(tmp_path / "texts"))
    run = RunResult(exit_code=0, hang=False, stdout="", stderr="", device_timing=timing())
    reference = stages._run_info(context, run, reference=True)
    assert reference.device_timing == timing(self_reported=False), "the baseline runs the item's tracked reference"
    attempt = stages._run_info(context, run)
    assert attempt.device_timing == timing(self_reported=True), "an attempt runs model-written code"
    plain = stages._run_info(context, RunResult(exit_code=0, hang=False, stdout="", stderr=""))
    assert plain.device_timing is None


def test_the_note_code_is_a_record_constant() -> None:
    assert record.DEVICE_TIMING_NOTE == "device-timing"


def test_trial_md_shows_the_reading_without_a_rate() -> None:
    from lassi.core import trial_md

    cell = trial_md._timing_cell(timing(programs=[program(), program(kernel_ns=3000, kernel_cycles=3000)]))
    assert cell == (
        "2 program(s): 10000 ns device kernel time, 10000 cycles at AICLK 1000 MHz (tt-metal device profiler, "
        "tt-metal@5280a9cf-tracy), self-reported"
    )
    assert trial_md._timing_cell(None) == "PLACEHOLDER"
    assert "/s" not in cell, "no rate is shown"


def test_parquet_holds_the_reading_as_json_text() -> None:
    import json

    from lassi.core import parquet
    from lassi.core.interfaces import Sampling
    from lassi.core.record import Attempt, BenchItem, ModelInfo, Provenance, Trial

    trial = Trial(
        trial_id="lassi-demo/arm/tt-pairs-v0/cpp-tt/loopback/run01",
        recipe_hash="0" * 64,
        provenance=Provenance(commit=None, dirty=None, device=None, sdk=None, date="2026-10-26T12:00:00+00:00"),
        bench_item=BenchItem(suite="tt-pairs-v0", item="loopback", split="unassigned", direction="cpp-tt"),
        model=ModelInfo(backend="none", id="none", sampling=Sampling(temperature=0.0, top_p=1.0, max_tokens=1)),
        attempts=[Attempt(index=0, stage_reached="S5", run=RunInfo(exit_code=0, hang=False, device_timing=timing()))],
    )
    rows = parquet.trial_rows([trial])
    assert rows["trials"][0]["reference_run_device_timing"] is None, "not measured"
    cell = rows["attempts"][0]["run_device_timing"]
    assert json.loads(cell) == record.to_dict(timing()) and cell == json.dumps(json.loads(cell), sort_keys=True)
