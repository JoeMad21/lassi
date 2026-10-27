"""Tests for the device-timing reading of silicon runs (task P17.42; plans/spikes/tt-perf.md, design (A)).

Bible: Execution Backends (the tt_silicon row), Result Record (RunInfo,
DeviceTiming), Evaluation Protocol (Reporting Rules), Agent Rules 1, 2, and
10.

lassi.executors.tt_profiler reads what tt-metal's device profiler writes in
a run of the Tracy-enabled build (toolchains/tt-metal.pin, BUILD tracy). At
tt-metal 5280a9cf it writes, under TT_METAL_PROFILER_DIR, a `.logs`
directory (tt_metal/impl/profiler/profiler_paths.hpp:16-39) holding
profile_log_device.csv, whose first line is `ARCH: <arch>, CHIP_FREQ[MHz]:
<aiclk>, Max Compute Cores: <n>` (profiler.cpp:1058-1066), and, with
TT_METAL_PROFILER_CPP_POST_PROCESS=1, cpp_device_perf_report.csv, one row
per program execution in the 28 columns profiler_analysis.cpp:586-601
writes from cpp_device_analyses.json's twelve analyses. These tests fix:

- REPORT_COLUMNS is that header, in order; profiler_environment(dir) is the
  four variables the executor adds with device_profiler kernel and nothing
  else of the profiler's (no NoC events, no perf counters, no sync).
- read_device_timing(dir, stderr, build) returns a DeviceTiming: the source
  "tt-metal device profiler", the build, the arch and AICLK of the device
  log's first line, one DeviceProgram per report row in file order
  (DEVICE ID, OP NAME, GLOBAL CALL COUNT, CORE COUNT, the kernel's cycles as
  END CYCLE minus START CYCLE, DEVICE KERNEL DURATION [ns], and DEVICE FW
  DURATION [ns] or None when empty), and self_reported True (a stage clears
  it for a tracked reference's run).
- Any shortfall raises TimingShortfall naming it, never a partial reading
  (Agent Rule 1): a missing file, another header, a row of another width or
  with a non-integer or empty kernel value, an end cycle before its start,
  no row, a device log without the clock line, an AICLK of 0 (what UMD
  reports for a simulated chip), and the profiler's dropped-markers warning
  in stderr.
- device_timing_reading(...) wraps it for the executor: (reading, None) on
  success; (None, note) otherwise, the note a run-stage Diagnostic of
  severity note and code device-timing that says why, including a run
  that did not end cleanly, which is not read at all.

Every CSV here is SYNTHETIC, written in the documented columns; no value is
a measurement, and nothing runs a program.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from lassi.core.record import DeviceProgram, DeviceTiming, Diagnostic

BUILD = "tt-metal@5280a9cf-tracy"
# The header profiler_analysis.cpp writes at 5280a9cf, with cpp_device_analyses.json's analyses in file order.
HEADER = (
    "GLOBAL CALL COUNT,METAL TRACE ID,METAL TRACE REPLAY SESSION ID,DEVICE ID,DEVICE ARCH,OP NAME,CORE COUNT,"
    "AVAILABLE WORKER CORE COUNT,DEVICE TRACE FIRMWARE DURATION [ns],DEVICE TRACE KERNEL DURATION [ns],"
    "DEVICE KERNEL FIRST TO LAST START [ns],DEVICE FW DURATION [ns],DEVICE FW START CYCLE,DEVICE FW END CYCLE,"
    "DEVICE KERNEL DURATION [ns],DEVICE KERNEL START CYCLE,DEVICE KERNEL END CYCLE,"
    "DEVICE KERNEL DURATION DM START [ns],DEVICE KERNEL DM START CYCLE,DEVICE KERNEL DM END CYCLE,"
    "DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],DEVICE TRISC0 KERNEL DURATION [ns],"
    "DEVICE TRISC1 KERNEL DURATION [ns],DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC KERNEL DURATION [ns],"
    "OP TO OP LATENCY [ns],OP TO OP LATENCY BR/NRISC START [ns]"
)
CLOCK_LINE = "ARCH: wormhole_b0, CHIP_FREQ[MHz]: 1000, Max Compute Cores: 64"
LOG_HEADER = (
    "PCIe slot, core_x, core_y, RISC processor type, timer_id, time[cycles since reset], data, run host ID, trace id, "
    "trace id counter, zone name, type, source line, source file, meta data"
)
# SYNTHETIC rows: program 1 on chip 0 with 1 core, 7000 kernel cycles; program 2 with 64 cores and no FW value.
ROW_ONE = "1024,,,0,wormhole_b0,,1,64,,,,9100,100,9200,7000,1000,8000,6900,1010,7910,6800,6700,6600,6500,6400,,0,0"
ROW_TWO = (
    "2048,,,0,wormhole_b0,matmul,64,64,,,,,,,512000,20000,532000,511000,20100,531100,510000,509000,508000,507000,"
    "506000,,12000,11900"
)
LOG_ROW = "0,1,1,BRISC,1,1000,0,1024,,,BRISC-KERNEL,ZONE_START,1,k.cpp,"
DROPPED = "Profiler DRAM buffers were full, markers were dropped! device 0, worker core 1, 1, Risc BRISC"


@pytest.fixture(scope="module")
def module() -> ModuleType:
    """Import lassi.executors.tt_profiler, failing clearly while task P17.42 has not added it."""
    try:
        return importlib.import_module("lassi.executors.tt_profiler")
    except ModuleNotFoundError as error:
        pytest.fail(f"lassi.executors.tt_profiler does not exist yet (task P17.42): {error}")


def write_logs(directory: Path, report: str | None = None, log: str | None = None) -> Path:
    """Write the SYNTHETIC profiler files under <directory>/.logs; None means the default text, "" leaves it out."""
    logs = directory / ".logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = f"{HEADER}\n{ROW_ONE}\n{ROW_TWO}\n" if report is None else report
    log = f"{CLOCK_LINE}\n{LOG_HEADER}\n{LOG_ROW}\n" if log is None else log
    if report:
        (logs / "cpp_device_perf_report.csv").write_bytes(report.encode("ascii"))
    if log:
        (logs / "profile_log_device.csv").write_bytes(log.encode("ascii"))
    return directory


def test_the_report_columns_are_the_ones_tt_metal_writes_at_the_pin(module: ModuleType) -> None:
    assert ",".join(module.REPORT_COLUMNS) == HEADER
    assert len(module.REPORT_COLUMNS) == 28
    assert module.LOGS_DIR == ".logs"
    assert (module.REPORT_FILE, module.DEVICE_LOG_FILE) == ("cpp_device_perf_report.csv", "profile_log_device.csv")
    assert module.SETTINGS == ("off", "kernel")


def test_the_profiler_environment_is_the_four_variables_and_nothing_else(module: ModuleType, tmp_path: Path) -> None:
    found = module.profiler_environment(tmp_path / "profiler")
    assert found == {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY": "1",
        "TT_METAL_PROFILER_DIR": str(tmp_path / "profiler"),
    }


def test_a_complete_report_gives_one_program_per_row_in_file_order(module: ModuleType, tmp_path: Path) -> None:
    timing = module.read_device_timing(write_logs(tmp_path / "profiler"), "", BUILD)
    assert isinstance(timing, DeviceTiming)
    assert (timing.source, timing.build, timing.arch, timing.aiclk_mhz) == (
        "tt-metal device profiler", BUILD, "wormhole_b0", 1000,
    )
    assert timing.self_reported is True, "the executor cannot tell a reference from model output; a stage clears it"
    assert timing.programs == [
        DeviceProgram(device_id=0, op_name="", call_count=1024, cores=1, kernel_cycles=7000, kernel_ns=7000,
                      fw_ns=9100),
        DeviceProgram(device_id=0, op_name="matmul", call_count=2048, cores=64, kernel_cycles=512000,
                      kernel_ns=512000, fw_ns=None),
    ]


@pytest.mark.parametrize(
    ("report", "log", "stderr", "named"),
    [
        pytest.param("", None, "", "cpp_device_perf_report.csv", id="no-report"),
        pytest.param(None, "", "", "profile_log_device.csv", id="no-device-log"),
        pytest.param(f"{HEADER.replace('OP NAME', 'OP')}\n{ROW_ONE}\n", None, "", "columns", id="other-header"),
        pytest.param(f"{HEADER}\n", None, "", "no program", id="no-row"),
        pytest.param(f"{HEADER}\n{ROW_ONE},extra\n", None, "", "28", id="wide-row"),
        pytest.param(f"{HEADER}\n{ROW_ONE.replace(',7000,1000,8000,', ',,1000,8000,')}\n", None, "",
                     "DEVICE KERNEL DURATION [ns]", id="empty-kernel-duration"),
        pytest.param(f"{HEADER}\n{ROW_ONE.replace(',7000,1000,8000,', ',7000,9000,8000,')}\n", None, "",
                     "before", id="end-before-start"),
        pytest.param(f"{HEADER}\n{ROW_ONE.replace('1024,', 'x1024,', 1)}\n", None, "", "GLOBAL CALL COUNT",
                     id="non-integer"),
        pytest.param(None, f"{LOG_HEADER}\n", "", "CHIP_FREQ", id="no-clock-line"),
        pytest.param(None, f"{CLOCK_LINE.replace(': 1000,', ': 0,')}\n{LOG_HEADER}\n", "", "AICLK", id="aiclk-0"),
        pytest.param(None, None, f"info\n{DROPPED}\n", "dropped", id="dropped-markers"),
    ],
)
def test_any_shortfall_is_refused_whole(
    module: ModuleType, tmp_path: Path, report: str | None, log: str | None, stderr: str, named: str
) -> None:
    directory = write_logs(tmp_path / "profiler", report, log)
    with pytest.raises(module.TimingShortfall, match=named.replace("[", r"\[").replace("]", r"\]")):
        module.read_device_timing(directory, stderr, BUILD)


def test_a_missing_directory_is_a_shortfall(module: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(module.TimingShortfall, match="cpp_device_perf_report.csv"):
        module.read_device_timing(tmp_path / "absent", "", BUILD)


def test_the_reading_gives_the_timing_or_a_run_note_naming_the_shortfall(module: ModuleType, tmp_path: Path) -> None:
    directory = write_logs(tmp_path / "profiler")
    timing, note = module.device_timing_reading(directory, exit_code=0, hang=False, stderr="", build=BUILD)
    assert isinstance(timing, DeviceTiming) and note is None
    timing, note = module.device_timing_reading(directory, exit_code=0, hang=False, stderr=DROPPED, build=BUILD)
    assert timing is None and isinstance(note, Diagnostic)
    assert (note.stage, note.severity, note.code) == ("run", "note", "device-timing")
    assert "dropped" in note.message and "not recorded" in note.message
    assert note.message.isascii()


@pytest.mark.parametrize(("exit_code", "hang"), [(1, False), (134, False), (0, True), (-1, False)])
def test_a_run_that_did_not_end_cleanly_is_not_read(
    module: ModuleType, tmp_path: Path, exit_code: int, hang: bool
) -> None:
    directory = write_logs(tmp_path / "profiler")
    timing, note = module.device_timing_reading(directory, exit_code=exit_code, hang=hang, stderr="", build=BUILD)
    assert timing is None and note is not None
    assert "did not end cleanly" in note.message
    assert str(exit_code) in note.message or "hang" in note.message


def test_the_dropped_markers_warning_on_stdout_is_a_shortfall(tmp_path: Path) -> None:
    # tt-metal's logger writes its warnings to stdout (the P4.9 captures), so the check reads both streams.
    tt_profiler = importlib.import_module("lassi.executors.tt_profiler")
    timing, note = tt_profiler.device_timing_reading(
        tmp_path, exit_code=0, hang=False, stderr="", stdout=f"info\n{DROPPED}\n", build="tt-metal@5280a9cf-tracy")
    assert timing is None and note is not None and "dropped" in note.message


def test_a_truncated_stream_is_a_shortfall(tmp_path: Path) -> None:
    tt_profiler = importlib.import_module("lassi.executors.tt_profiler")
    timing, note = tt_profiler.device_timing_reading(
        tmp_path, exit_code=0, hang=False, stderr="", stdout="", build="tt-metal@5280a9cf-tracy", truncated=True)
    assert timing is None and note is not None and "truncated" in note.message
