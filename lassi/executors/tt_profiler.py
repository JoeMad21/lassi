"""Device time of a silicon run from tt-metal's device profiler (task P17.42; plans/spikes/tt-perf.md, design (A)).

Bible: Execution Backends (the tt_silicon row), Result Record (RunInfo,
DeviceTiming), Evaluation Protocol (Reporting Rules), Agent Rules 1, 2, and
10. The owner's answer of 2026-10-26 to OQ-105, option (b):
the main tt-metal pin is built with Tracy (toolchains/tt-metal.pin, BUILD
tracy), the build in which tt-metal compiles its device profiler.

The tt_silicon executor (lassi.executors.tt_silicon) takes the setting
device_profiler, one of SETTINGS, with no default. With "kernel" a run's
environment gains profiler_environment(<workdir>/@tt-silicon/profiler) and
nothing else of the profiler's: the device profiler, its C++ post-processing
(which writes the per-program report), no push to a Tracy server, and the
output directory. No NoC events, perf counters, host-device sync, or
dispatch-core profiling. The ttsim executor never sets these (ttsim reports
an AICLK of 0 and its clock follows the host's reads; Agent Rule 2), and the
sandbox refuses them without a Tenstorrent exposure.

What tt-metal 5280a9cf writes, read at the pin (tt_metal/impl/profiler/):
under TT_METAL_PROFILER_DIR, a directory LOGS_DIR (profiler_paths.hpp:16-39)
holding DEVICE_LOG_FILE, whose first line is `ARCH: <arch>, CHIP_FREQ[MHz]:
<aiclk>, Max Compute Cores: <n>` (profiler.cpp:1058-1066; the AICLK the card
reported through UMD when the device opened), and REPORT_FILE, one row per
program execution in REPORT_COLUMNS (profiler_analysis.cpp:586-601, with
tt_metal/tools/profiler/cpp_device_analyses.json's twelve analyses in file
order). DEVICE KERNEL DURATION [ns] spans the first kernel start to the
last kernel end over every core and RISC of the program, its START and END
CYCLE the core wall clock's cycles; tt-metal converts at the AICLK it reads
(profiler_analysis.cpp:270-276). When the DRAM buffers fill, markers are
dropped and tt-metal logs DROPPED (profiler.cpp:1480).

read_device_timing(directory, stderr, build) parses them into a
lassi.core.record DeviceTiming or raises TimingShortfall naming what is
missing or wrong: a missing file, another header, a row of another width or
with a value that is not a whole number where one is required, an end cycle
before its start, no row, a device log without its clock line, an AICLK of
0, or the dropped-markers warning in stderr. Nothing partial is ever
returned (Agent Rule 1). device_timing_reading(...) wraps it for the
executor: a run that did not end cleanly is not read, and any shortfall
becomes one run-stage note coded DEVICE_TIMING_NOTE. The reading's
self_reported is True: the executor cannot tell a tracked reference from a
model's program, and a stage clears it for the item's reference.

Unverified until the Tracy build and an owner-window run on the cards: that
the five tracked references are profiled at mesh device close without a
code change, that the files are written inside the sandbox (read-only tree,
empty network namespace), the AICLK the cards report, and the per-marker
overhead. The parsing here is tested on SYNTHETIC files in the documented
columns.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

from lassi.core.record import DEVICE_TIMING_NOTE, DeviceProgram, DeviceTiming, Diagnostic

# The device_profiler setting's values, each named, with no default.
SETTINGS = ("off", "kernel")
# tt-metal's directory and file names under TT_METAL_PROFILER_DIR (profiler_paths.hpp:12, 17, 18 at 5280a9cf).
LOGS_DIR = ".logs"
REPORT_FILE = "cpp_device_perf_report.csv"
DEVICE_LOG_FILE = "profile_log_device.csv"
SOURCE = "tt-metal device profiler"
# The per-program report's header at 5280a9cf: eight program columns, then each analysis of cpp_device_analyses.json
# (its start and end cycle columns after those that display them), then the two latency columns.
REPORT_COLUMNS = (
    "GLOBAL CALL COUNT",
    "METAL TRACE ID",
    "METAL TRACE REPLAY SESSION ID",
    "DEVICE ID",
    "DEVICE ARCH",
    "OP NAME",
    "CORE COUNT",
    "AVAILABLE WORKER CORE COUNT",
    "DEVICE TRACE FIRMWARE DURATION [ns]",
    "DEVICE TRACE KERNEL DURATION [ns]",
    "DEVICE KERNEL FIRST TO LAST START [ns]",
    "DEVICE FW DURATION [ns]",
    "DEVICE FW START CYCLE",
    "DEVICE FW END CYCLE",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE KERNEL START CYCLE",
    "DEVICE KERNEL END CYCLE",
    "DEVICE KERNEL DURATION DM START [ns]",
    "DEVICE KERNEL DM START CYCLE",
    "DEVICE KERNEL DM END CYCLE",
    "DEVICE BRISC KERNEL DURATION [ns]",
    "DEVICE NCRISC KERNEL DURATION [ns]",
    "DEVICE TRISC0 KERNEL DURATION [ns]",
    "DEVICE TRISC1 KERNEL DURATION [ns]",
    "DEVICE TRISC2 KERNEL DURATION [ns]",
    "DEVICE ERISC KERNEL DURATION [ns]",
    "OP TO OP LATENCY [ns]",
    "OP TO OP LATENCY BR/NRISC START [ns]",
)
# The columns a reading needs whole in every row; DEVICE FW DURATION [ns] may be empty.
REQUIRED = (
    "GLOBAL CALL COUNT",
    "DEVICE ID",
    "CORE COUNT",
    "DEVICE KERNEL DURATION [ns]",
    "DEVICE KERNEL START CYCLE",
    "DEVICE KERNEL END CYCLE",
)
FW_COLUMN = "DEVICE FW DURATION [ns]"
# The warning tt-metal logs when the profiler's DRAM buffers filled and markers were lost (profiler.cpp:1480).
DROPPED = "Profiler DRAM buffers were full, markers were dropped!"
# The device log's first line (profiler.cpp:1060-1062).
_CLOCK_LINE = re.compile(r"ARCH: (\S+), CHIP_FREQ\[MHz\]: (\d+), Max Compute Cores: (\d+)")
_WHOLE = re.compile(r"\d+")


class TimingShortfall(ValueError):
    """The profiler's files cannot give a whole reading; the message names what is missing or wrong."""


def profiler_environment(directory: Path) -> dict[str, str]:
    """Return the four variables a run with device_profiler kernel adds, writing under `directory`."""
    return {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        "TT_METAL_PROFILER_DISABLE_PUSH_TO_TRACY": "1",
        "TT_METAL_PROFILER_DIR": str(directory),
    }


def _text(path: Path) -> str:
    """Return a profiler file's text as ASCII (other bytes replaced); TimingShortfall naming it when it is missing."""
    try:
        return path.read_bytes().decode("ascii", errors="replace")
    except OSError as error:
        raise TimingShortfall(f"{path.name} cannot be read under {path.parent} ({type(error).__name__})") from None


def read_clock(text: str) -> tuple[str, int]:
    """Return (arch, AICLK in MHz) from the device log's first line; TimingShortfall when it is not the clock line."""
    first = text.splitlines()[0] if text else ""
    found = _CLOCK_LINE.fullmatch(first.strip())
    if found is None:
        raise TimingShortfall(f"{DEVICE_LOG_FILE}'s first line is not the ARCH, CHIP_FREQ[MHz] line: {first[:120]!r}")
    aiclk = int(found.group(2))
    if aiclk < 1:
        raise TimingShortfall(f"{DEVICE_LOG_FILE} gives an AICLK of {aiclk} MHz, as UMD reports for a simulated chip")
    return found.group(1), aiclk


def _whole(row: dict[str, str], column: str, number: int) -> int:
    """Return one cell as a whole number; TimingShortfall naming the row and column when it is empty or not one."""
    value = row[column].strip()
    if not _WHOLE.fullmatch(value):
        raise TimingShortfall(f"{REPORT_FILE} row {number}: {column} is {value!r}, not a whole number")
    return int(value)


def _program(row: dict[str, str], number: int) -> DeviceProgram:
    """Return one report row as a DeviceProgram; TimingShortfall for a value it cannot take."""
    values = {column: _whole(row, column, number) for column in REQUIRED}
    start, end = values["DEVICE KERNEL START CYCLE"], values["DEVICE KERNEL END CYCLE"]
    if end <= start:
        raise TimingShortfall(f"{REPORT_FILE} row {number}: the kernel's end cycle {end} is at or before its "
                              f"start {start}")
    fw = row[FW_COLUMN].strip()
    return DeviceProgram(
        device_id=values["DEVICE ID"],
        op_name=row["OP NAME"].strip(),
        call_count=values["GLOBAL CALL COUNT"],
        cores=values["CORE COUNT"],
        kernel_cycles=end - start,
        kernel_ns=values["DEVICE KERNEL DURATION [ns]"],
        fw_ns=_whole(row, FW_COLUMN, number) if fw else None,
    )


def read_report(text: str) -> list[DeviceProgram]:
    """Return one DeviceProgram per row of the per-program report, in file order; TimingShortfall on any shortfall."""
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or tuple(cell.strip() for cell in rows[0]) != REPORT_COLUMNS:
        raise TimingShortfall(f"{REPORT_FILE}'s columns are not the ones tt-metal 5280a9cf writes")
    programs = []
    for number, cells in enumerate(rows[1:], start=1):
        if not cells:
            continue
        if len(cells) != len(REPORT_COLUMNS):
            raise TimingShortfall(f"{REPORT_FILE} row {number} has {len(cells)} cells, not {len(REPORT_COLUMNS)}")
        programs.append(_program(dict(zip(REPORT_COLUMNS, cells, strict=True)), number))
    if not programs:
        raise TimingShortfall(f"{REPORT_FILE} holds no program row")
    return programs


def read_device_timing(directory: Path, stderr: str, build: str, stdout: str = "") -> DeviceTiming:
    """Return the run's DeviceTiming from the profiler's files under `directory`; TimingShortfall on any shortfall.

    `build` names the tt-metal install the run used (the pin's PREFIX_NAME).
    The dropped-markers warning in `stdout` or `stderr` is a shortfall (tt-metal's
    logger writes its warnings to stdout), since a program's first or last
    marker may be among those lost.
    """
    if DROPPED in stdout or DROPPED in stderr:
        raise TimingShortfall("tt-metal logged that profiler markers were dropped (its DRAM buffers were full)")
    logs = Path(directory) / LOGS_DIR
    programs = read_report(_text(logs / REPORT_FILE))
    arch, aiclk = read_clock(_text(logs / DEVICE_LOG_FILE))
    return DeviceTiming(source=SOURCE, build=build, arch=arch, aiclk_mhz=aiclk, programs=programs, self_reported=True)


def _note(why: str) -> Diagnostic:
    """Return the run-stage note that says why the device time was not recorded."""
    message = f"device timing not recorded: {why} ({SOURCE}; RunInfo.device_timing stays null, never partial)"
    text = message.encode("ascii", errors="replace").decode("ascii")
    return Diagnostic(stage="run", severity="note", code=DEVICE_TIMING_NOTE, message=text)


def device_timing_reading(
    directory: Path, *, exit_code: int | None, hang: bool, stderr: str, build: str, stdout: str = "",
    truncated: bool = False,
) -> tuple[DeviceTiming | None, Diagnostic | None]:
    """Return (reading, None), or (None, note) for a run that did not end cleanly or any shortfall.

    A clean ending is exit status 0 with no hang: the profiler reads the
    device and writes its files when the program closes its mesh device, so
    any other ending leaves them absent or partial, and they are not read.
    """
    if hang or exit_code != 0:
        ending = "a hang" if hang else f"exit status {exit_code}"
        return None, _note(f"the run did not end cleanly ({ending}), so the profiler's files are not read")
    if truncated:
        return None, _note("the run's output was truncated, so a dropped-markers warning may be missing")
    try:
        return read_device_timing(directory, stderr, build, stdout), None
    except TimingShortfall as error:
        return None, _note(str(error))
