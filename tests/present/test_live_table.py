"""Tests for the live inference table: its pure rendering and the observer that redraws it (task P4.8).

Bible: Readability Standards, Terminal Presentation (the Inference table
bullet: a boxed table in the style of `furiosa-smi status`, `+---+` rules
and `|` cells, under a header that names the run, the arm, the device, the
elapsed time, and the trial's place in the run; columns Step, Status, and
Code; rows for the source program, each attempt in order, the latest code
while the model works, and the final product with its end reason and
correction count; a code cell holds a bounded excerpt, about 12 lines from
the first kernel, launch, or pragma line, cut to the terminal width; no
Code column without an interactive terminal; and the Rules: plain ASCII
glyphs, ANSI escape sequences only on a terminal, source and model text
only as bounded excerpts); Design Principle 2 (the table reads the one
record, so a Status cell shows the record's own stage and end reason).
plans/p4-ttsim.md, the planning decision "Progress hook (P4.8)".

The contract these tests fix, in lassi/present/live.py:

- TableHeader(run, arm, device, elapsed_s, number, count): the header
  values. The header shows the elapsed time as H:MM:SS (3725.4 s is
  1:02:05) and the place as `<number> of <count>` or `<number>/<count>`.
- render_table(trial, header, *, source, width, terminal, working=False,
  ended=False) -> str: one frame, a pure function of its arguments (no
  clock, no file, no terminal query). `source` is the starter code (file
  name -> text), `working` says a model request is out, and `ended` that
  the trial has ended. The frame is header lines, then the box: a rule, the
  column row, a rule, then each row followed by a rule. Every rule is `+`
  joined by `-` (or `=`) runs, and every box line has `|` exactly where the
  rules have `+`. The rows, in order: the source program (its Step names
  the source); one row per attempt (Step `translation` for attempt 0 and
  `correction <k>` for attempt k; Status shows the attempt's stage code, S0
  to S5, and its first error's message); with `working`, the latest code
  row; with `ended`, the final row (Step names the final product; Step and
  Status show the final stage when there is one, the end reason's code when
  there is one, and the correction count beside the word correction). On a
  terminal the Code column holds, for the source row, the latest code row
  (the last attempt's file), and the final row (the last attempt's file),
  the excerpt: 12 lines from the first line holding a
  kernel (__global__), a launch (<<<), or a pragma (#pragma), else from the
  first line, each line cut to fit the width, optionally after one label
  line naming the file. Without a terminal there is no Code column and no
  code text anywhere. Every line fits `width`. After removing the CSI
  sequences a terminal frame may use for color, every character is
  printable ASCII or a line feed; without a terminal no escape character
  appears at all, and no control sequence from the code ever reaches the
  output.
- clock: the module's monotonic clock in seconds, read at call time, which
  a test can patch; LiveTable(stream, *, terminal, width, clock=None) reads
  it when no clock is given.
- LiveTable is the observer `lassi run` passes with graphics on. Called
  with a ProgressEvent (lassi.core.progress), it keeps the latest trial
  snapshot and draws a frame to `stream` when the event is the first it
  gets, when it is a trial-end event (the trial's final frame), or when
  about a second (1.0 s) has passed on its clock since its last
  draw; other events draw nothing. tick() draws when about a second has
  passed since the last draw, and draws nothing before the first event. The
  header's run id, place, and source come from the trial's trial-start
  event, the arm from Trial.model.id, the device from
  Trial.provenance.device, and the elapsed time counts from the first
  event. Between a request-sent event and the next attempt event the frame
  shows the latest code row. On a terminal each frame after the first is
  written with escape sequences that redraw it in place; without one, no
  escape character is written and each trial's final frame still is (how
  often other frames are written there is left open).
- Added with the P4.8 commit audit: render_table(..., height=None) fits a
  taller frame to `height` lines, cutting the code excerpts first, then
  leaving out the oldest attempt rows, with a header line that says so, and
  keeps the top of a frame that still does not fit with a last line saying
  it was cut; LiveTable(..., height=None) fits its terminal frames to it.
  The excerpt's anchor is `__global__`, `__kernel`, or `kernel_main` as a
  whole word, `<<<`, or `#pragma` as the first text on a line, and trailing
  blank lines are dropped. A LiveTable that raised never draws again, and a
  tick that raises in ticking() stops the ticks with one stderr line naming
  the exception's class.
- Added with the P4.8 recheck: a LiveTable whose write to its stream
  raises OSError calls lassi.core.progress.mute_stderr on it, which points
  the file descriptor under it at the null device only when the stream is
  sys.stderr (the test's streams report descriptors of the test's own
  files, so the test process's stderr is never touched).

Snapshots: each scenario's frame, on a terminal and without one, at width
100, is compared with tests/present/golden/live-table/<name>.txt (escape
characters written as the four characters \\x1b). A missing golden file
fails the test; LASSI_RECORD_GOLDEN=1 writes it from the current rendering
and skips, for review before it is committed. Every trial, file, and
header value here is SYNTHETIC; no value is a measurement.
"""

from __future__ import annotations

import copy
import importlib
import io
import os
import re
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from lassi.core.interfaces import Sampling
from lassi.core.record import (
    Attempt,
    BenchItem,
    Diagnostic,
    EndReason,
    Final,
    ModelInfo,
    Provenance,
    RunInfo,
    Trial,
)

GOLDEN = Path(__file__).resolve().parent / "golden" / "live-table"
RECORD_VARIABLE = "LASSI_RECORD_GOLDEN"
EXCERPT_LINES = 12

RUN = "p48-synthetic-run"
ARM = "synthetic-arm-7b"
DEVICE = "SYNTHETIC device of the live table tests"
TRIAL_ID = f"p48-live/{ARM}/lassi-hecbench-10/omp-cuda/layout/run02"
ELAPSED_S = 3725.4
ELAPSED_TEXT = re.compile(r"(?<![0-9:])0?1:02:05(?![0-9])")
PLACE_TEXT = re.compile(r"(?<![0-9])2\s*(?:/|of)\s*5(?![0-9])")

# The SYNTHETIC starter program: its pragma line (line 10, 83 characters) is the excerpt's first line.
SOURCE_LINES = (
    "// SYNTHETIC starter program of the live table tests",
    "#include <cstdio>",
    "#include <vector>",
    "",
    "static const int N = 1024;",
    "",
    "int main() {",
    "  std::vector<float> a(N), b(N);",
    "  for (int i = 0; i < N; ++i) a[i] = i;",
    "#pragma omp target teams distribute parallel for map(to: a[0:N]) map(from: b[0:N])",
    "  for (int i = 0; i < N; ++i) {",
    "    b[i] = 2.0f * a[i];",
    "  }",
    "  float sum = 0.0f;",
    "  for (int i = 0; i < N; ++i) {",
    "    sum += b[i];",
    "  }",
    "  if (sum < 0.0f) {",
    "    std::puts(\"FAIL\");",
    "    return 1;",
    "  }",
    "  std::puts(\"PASS\");",
    "  return 0;",
    "}",
)
SOURCE_TEXT = "\n".join(SOURCE_LINES) + "\n"
SOURCE = {"main.cpp": SOURCE_TEXT}
SOURCE_ANCHOR = 9
NO_ANCHOR_TEXT = "int main() {\n  return 0;\n}\n"


def cuda_lines(version: int, broken: bool = False) -> tuple[str, ...]:
    """Return a SYNTHETIC CUDA translation whose kernel line names `version`; `broken` misspells a member."""
    member = "blockIdx.xx" if broken else "blockIdx.x"
    return (
        "// SYNTHETIC translation of the live table tests",
        "#include <cstdio>",
        "",
        "static const int N = 1024;",
        "",
        f"__global__ void scale_v{version}(const float* a, float* b) {{",
        f"  int i = {member} * blockDim.x + threadIdx.x;",
        "  if (i < N) b[i] = 2.0f * a[i];",
        "}",
        "",
        "int main() {",
        "  float *da, *db;",
        "  cudaMalloc(&da, N * sizeof(float));",
        "  cudaMalloc(&db, N * sizeof(float));",
        f"  scale_v{version}<<<N / 256, 256>>>(da, db);",
        "  cudaDeviceSynchronize();",
        "  cudaFree(da);",
        "  cudaFree(db);",
        "  std::puts(\"PASS\");",
        "  return 0;",
        "}",
    )


CUDA_ANCHOR = 5
WARNING = Diagnostic(stage="compile", severity="warning", code="W1", file="main.cu", line=4, message="unused N")
COMPILE_ERROR = Diagnostic(
    stage="compile", severity="error", code="E1", file="main.cu", line=7, column=11, message="no member named xx"
)
RUN_ERROR = Diagnostic(stage="run", severity="error", code="run-error", message="exit status 1")


def attempt(index: int, stage: str, *, broken: bool = False, diagnostics: Sequence[Diagnostic] = (),
            exit_code: int | None = None, text: str | None = None) -> Attempt:
    """Return a SYNTHETIC attempt with one file, main.cu; a run is recorded when `exit_code` is given."""
    files = {"main.cu": text if text is not None else "\n".join(cuda_lines(index, broken)) + "\n"}
    run = RunInfo() if exit_code is None else RunInfo(exit_code=exit_code, hang=False)
    return Attempt(index=index, files=files, stage_reached=stage, diagnostics=list(diagnostics), run=run)


def trial(attempts: Sequence[Attempt], final: Final | None = None) -> Trial:
    """Return a SYNTHETIC trial with `attempts` and `final` (the default Final while the trial runs)."""
    return Trial(
        trial_id=TRIAL_ID,
        recipe_hash="a" * 64,
        provenance=Provenance(commit=None, dirty=None, device=DEVICE, sdk=None, date="2026-09-26T00:00:00+00:00"),
        bench_item=BenchItem(suite="lassi-hecbench-10", item="layout", split="eval", direction="omp-cuda"),
        model=ModelInfo(backend="mock", id=ARM, sampling=Sampling(temperature=0.2, top_p=0.9, max_tokens=4096)),
        requests=[],
        attempts=list(attempts),
        final=final or Final(),
    )


CAP_MESSAGE = "the correction cap of 3 was reached with an error remaining"
SCENARIOS: dict[str, Trial] = {
    "clean-pass": trial([attempt(0, "S5", exit_code=0)], Final(stage_reached="S5", corrections=0)),
    "corrections": trial(
        [
            attempt(0, "S1", broken=True, diagnostics=[WARNING, COMPILE_ERROR]),
            attempt(1, "S4", diagnostics=[RUN_ERROR], exit_code=1),
            attempt(2, "S5", exit_code=0),
        ],
        Final(stage_reached="S5", corrections=2),
    ),
    "cap-hit": trial(
        [attempt(k, "S1", broken=True, diagnostics=[COMPILE_ERROR]) for k in range(4)],
        Final(stage_reached="S1", corrections=3, end_reason=EndReason(code="correction-cap", message=CAP_MESSAGE)),
    ),
    "baseline-end": trial(
        [],
        Final(corrections=0, end_reason=EndReason(code="baseline-run", message="the cuda reference exited with 1")),
    ),
}


# ---------------------------------------------------------------------------
# Modules under test, imported per test so each test fails on its own


def live() -> ModuleType:
    """Import and return lassi.present.live."""
    return importlib.import_module("lassi.present.live")


def progress() -> ModuleType:
    """Import and return lassi.core.progress."""
    return importlib.import_module("lassi.core.progress")


def header(**changes: Any) -> Any:
    """Return the SYNTHETIC TableHeader: run 2 of 5, 3725.4 s elapsed."""
    values: dict[str, Any] = {"run": RUN, "arm": ARM, "device": DEVICE, "elapsed_s": ELAPSED_S, "number": 2,
                              "count": 5}
    values.update(changes)
    return live().TableHeader(**values)


def render(case: Trial, *, terminal: bool = True, width: int = 120, working: bool = False, ended: bool = True,
           source: Mapping[str, str] = SOURCE) -> str:
    """Render `case` with the SYNTHETIC header and source."""
    return live().render_table(case, header(), source=source, width=width, terminal=terminal, working=working,
                               ended=ended)


# ---------------------------------------------------------------------------
# Reading a frame back


_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OTHER_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_RULE = re.compile(r"^\+(?:[-=]+\+)+$")
_COLUMN_ROW = re.compile(r"^\|\s*Step\s*\|\s*Status\s*\|(?:\s*Code\s*\|)?$")


def visible(text: str) -> str:
    """Return `text` without escape sequences and carriage returns: what a person reads."""
    return _OTHER_ESCAPES.sub("", _CSI.sub("", text)).replace("\r", "")


@dataclass
class Row:
    """One table row: each column's lines, as drawn."""

    cells: list[list[str]]

    def text(self, column: int) -> str:
        """Return one column's lines joined by spaces, each whitespace run written as one space."""
        return " ".join(" ".join(self.cells[column]).split())

    def code(self) -> list[str]:
        """Return the Code column's lines, each with its whitespace runs written as one space, trailing blanks cut."""
        lines = [" ".join(line.split()) for line in self.cells[2]]
        while lines and not lines[-1]:
            lines.pop()
        return lines


@dataclass
class Frame:
    """One frame read back: its header lines, its column names, and its rows."""

    header: list[str]
    columns: list[str]
    rows: list[Row]
    box: list[str]


def parse_frame(text: str) -> Frame:
    """Read one frame: header lines up to the first rule, then the box, closed by a rule after every row."""
    lines = visible(text).split("\n")
    start = next(index for index, line in enumerate(lines) if _RULE.match(line))
    bounds = [position for position, char in enumerate(lines[start]) if char == "+"]
    groups: list[list[str]] = []
    box: list[str] = []
    current: list[str] | None = None
    for line in lines[start:]:
        if _RULE.match(line):
            assert [p for p, c in enumerate(line) if c == "+"] == bounds, f"a rule does not match the box: {line!r}"
            if current is not None:
                groups.append(current)
            current = []
        elif line.startswith("|"):
            assert current is not None, "a box line outside the box"
            assert all(position < len(line) and line[position] == "|" for position in bounds), (
                f"a box line has no | where the rules have +: {line!r}"
            )
            current.append(line)
        else:
            break
        box.append(line)
    assert current == [], "the box does not end with a rule"
    rows = [Row([[line[bounds[i] + 1:bounds[i + 1]] for line in group] for i in range(len(bounds) - 1)])
            for group in groups]
    columns = [rows[0].text(column) for column in range(len(bounds) - 1)]
    return Frame(header=lines[:start], columns=columns, rows=rows[1:], box=box)


def excerpt(lines: Sequence[str], anchor: int) -> list[str]:
    """Return the expected excerpt: EXCERPT_LINES lines from `anchor`, whitespace collapsed, trailing blanks cut."""
    shown = [" ".join(line.split()) for line in lines[anchor:anchor + EXCERPT_LINES]]
    while shown and not shown[-1]:
        shown.pop()
    return shown


def shown_code(row: Row, lines: Sequence[str]) -> list[str]:
    """Return a row's code lines, dropping one leading label line that names the file and is not a code line."""
    code = row.code()
    known = {" ".join(line.split()) for line in lines}
    if code and code[0] not in known and re.search(r"main\.(?:cu|cpp)", code[0]):
        code = code[1:]
    return code


def cut_of(shown: str, line: str) -> bool:
    """Return True when `shown` is `line`, or a cut of it: a prefix of at least 10 characters, a marker allowed."""
    shown, line = " ".join(shown.split()), " ".join(line.split())
    if shown == line:
        return True
    kept = re.sub(r"[.~>+-]+$", "", shown).rstrip()
    return len(kept) >= min(10, len(line)) and len(kept) < len(line) and line.startswith(kept)


def squashed(text: str) -> str:
    """Return `text` without any whitespace, so a message wrapped over lines still matches."""
    return "".join(text.split())


# ---------------------------------------------------------------------------
# Columns and rows


@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_on_a_terminal_the_columns_are_step_status_and_code(name: str) -> None:
    assert parse_frame(render(SCENARIOS[name], terminal=True)).columns == ["Step", "Status", "Code"]


@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_without_a_terminal_there_is_no_code_column_and_no_code(name: str) -> None:
    text = render(SCENARIOS[name], terminal=False)
    assert parse_frame(text).columns == ["Step", "Status"]
    assert "\x1b" not in text, "no escape sequence without a terminal"
    for fragment in ("#pragma omp", "__global__", "cudaMalloc", "std::vector"):
        assert fragment not in text, f"code text {fragment!r} shown without a terminal"


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_the_rows_are_the_source_each_attempt_in_order_and_the_final_product(name: str, terminal: bool) -> None:
    case = SCENARIOS[name]
    rows = parse_frame(render(case, terminal=terminal)).rows
    assert len(rows) == 1 + len(case.attempts) + 1
    assert "source" in rows[0].text(0).lower()
    for index, row in enumerate(rows[1:-1]):
        step = row.text(0).lower()
        if index == 0:
            assert "translation" in step, step
        else:
            assert "correction" in step and re.search(rf"(?<![0-9]){index}(?![0-9])", step), step
    assert "final" in rows[-1].text(0).lower()


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
@pytest.mark.parametrize("name", ["clean-pass", "corrections", "cap-hit"])
def test_each_attempt_row_shows_its_stage_and_its_first_error(name: str, terminal: bool) -> None:
    case = SCENARIOS[name]
    rows = parse_frame(render(case, terminal=terminal)).rows
    for row, shown in zip(rows[1:-1], case.attempts, strict=True):
        status = row.text(1)
        assert re.search(rf"\b{shown.stage_reached}\b", status), status
        errors = [item for item in shown.diagnostics if item.severity == "error"]
        if errors:
            assert squashed(errors[0].message) in squashed(status), status


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_the_final_row_shows_the_stage_the_end_reason_and_the_correction_count(name: str, terminal: bool) -> None:
    final = SCENARIOS[name].final
    row = parse_frame(render(SCENARIOS[name], terminal=terminal)).rows[-1]
    text = f"{row.text(0)} {row.text(1)}"
    if final.stage_reached is not None:
        assert re.search(rf"\b{final.stage_reached}\b", text), text
    if final.end_reason is not None:
        assert final.end_reason.code in text, text
    count = final.corrections
    beside = rf"(?<![0-9.S]){count}\s+corrections?\b|\bcorrections?\W{{0,3}}{count}(?![0-9.])"
    assert re.search(beside, text, re.IGNORECASE), f"the correction count {count} is not beside the word: {text!r}"


def test_while_the_model_works_the_latest_code_row_follows_the_attempts() -> None:
    case = trial([attempt(0, "S1", broken=True, diagnostics=[COMPILE_ERROR]), attempt(1, "S4", exit_code=1)])
    idle = parse_frame(render(case, working=False, ended=False)).rows
    working = parse_frame(render(case, working=True, ended=False)).rows
    assert len(idle) == 1 + 2, "no final row and no latest code row while the trial runs and no request is out"
    assert len(working) == 1 + 2 + 1, "a request is out: the latest code row follows the attempts"
    assert shown_code(working[-1], cuda_lines(1))[0] == " ".join(cuda_lines(1)[CUDA_ANCHOR].split())


def test_before_the_translation_arrives_the_table_shows_the_source_and_the_working_row() -> None:
    rows = parse_frame(render(trial([]), working=True, ended=False)).rows
    assert len(rows) == 2
    assert "source" in rows[0].text(0).lower()


# ---------------------------------------------------------------------------
# The code excerpt


def test_a_code_cell_shows_12_lines_from_the_first_kernel_launch_or_pragma_line() -> None:
    rows = parse_frame(render(SCENARIOS["corrections"], width=200)).rows
    assert shown_code(rows[0], SOURCE_LINES) == excerpt(SOURCE_LINES, SOURCE_ANCHOR), "the source row"
    last = cuda_lines(2)
    assert shown_code(rows[-1], last) == excerpt(last, CUDA_ANCHOR), "the final row shows the last attempt"


def test_a_file_with_no_kernel_launch_or_pragma_line_shows_its_first_lines() -> None:
    case = trial([attempt(0, "S4", text=NO_ANCHOR_TEXT)], Final(stage_reached="S4", corrections=0))
    lines = NO_ANCHOR_TEXT.rstrip("\n").split("\n")
    rows = parse_frame(render(case, width=200)).rows
    assert shown_code(rows[-1], lines) == [" ".join(line.split()) for line in lines]


def test_a_code_line_is_cut_to_the_width() -> None:
    text = render(SCENARIOS["clean-pass"], width=80)
    assert all(len(line) <= 80 for line in visible(text).split("\n"))
    first = shown_code(parse_frame(text).rows[0], SOURCE_LINES)[0]
    assert cut_of(first, SOURCE_LINES[SOURCE_ANCHOR]), first


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
@pytest.mark.parametrize("width", [60, 80, 100, 132])
@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_every_line_fits_the_width(name: str, width: int, terminal: bool) -> None:
    text = render(SCENARIOS[name], terminal=terminal, width=width)
    too_wide = [line for line in visible(text).split("\n") if len(line) > width]
    assert not too_wide, f"{len(too_wide)} line(s) wider than {width}: {too_wide[:2]}"


# ---------------------------------------------------------------------------
# The header


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
def test_the_header_names_the_run_the_arm_the_device_the_elapsed_time_and_the_place(terminal: bool) -> None:
    head = " ".join(parse_frame(render(SCENARIOS["corrections"], terminal=terminal)).header)
    for value in (RUN, ARM, DEVICE):
        assert value in head, f"{value!r} is not in the header: {head!r}"
    assert ELAPSED_TEXT.search(head), f"the elapsed time 1:02:05 is not in the header: {head!r}"
    assert PLACE_TEXT.search(head), f"the place 2 of 5 is not in the header: {head!r}"


# ---------------------------------------------------------------------------
# Glyphs and escape sequences


# SYNTHETIC hostile code, every line inside the excerpt: a title change, a clear, a reset, a tab, two characters
# beyond ASCII, and a carriage return.
HOSTILE_TEXT = (
    "#pragma omp target // a title change \x1b]0;pwned\x07, a clear \x1b[2J, a reset \x1bc\n"
    f"\t// tab, caf{chr(0xE9)}, dash {chr(0x2014)} and a carriage return\r\n"
    "int main() { return 0; }\n"
)
HOSTILE_SOURCE = {"main.cpp": HOSTILE_TEXT}
HOSTILE_ERROR = Diagnostic(stage="compile", severity="error", message="bad \x1b]0;pwned\x07 token \x1b[2J\r")


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_every_glyph_is_plain_ascii(name: str, terminal: bool) -> None:
    text = render(SCENARIOS[name], terminal=terminal, source=HOSTILE_SOURCE)
    if not terminal:
        assert "\x1b" not in text
    plain = _CSI.sub("", text)
    odd = sorted({char for char in plain if not (" " <= char <= "~" or char == "\n")})
    assert not odd, f"characters other than printable ASCII and line feeds: {odd!r}"


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "no-terminal"])
def test_no_control_sequence_from_the_code_or_a_message_reaches_the_output(terminal: bool) -> None:
    failed = attempt(0, "S1", text=HOSTILE_TEXT, diagnostics=[HOSTILE_ERROR])
    case = trial([failed, attempt(1, "S4", text=HOSTILE_TEXT)], Final(stage_reached="S4", corrections=1))
    text = render(case, terminal=terminal, source=HOSTILE_SOURCE)
    for sequence in ("\x1b]", "\x07", "\x1b[2J", "\x1bc", "\r"):
        assert sequence not in text, f"{sequence!r} from the code or a message reached the output"
    plain = _CSI.sub("", text)
    assert "\x1b" not in plain, "an escape character outside the frame's own color sequences"


# ---------------------------------------------------------------------------
# Purity


def test_rendering_is_pure() -> None:
    case = SCENARIOS["corrections"]
    before = copy.deepcopy(case)
    first = render(case, terminal=True)
    second = render(case, terminal=True)
    assert first == second, "the same snapshot and header give the same frame"
    assert case == before, "rendering never changes the trial"


# ---------------------------------------------------------------------------
# Snapshots


def _golden(name: str, text: str) -> None:
    """Compare `text` with the golden file `name`; record it and skip when LASSI_RECORD_GOLDEN=1."""
    path = GOLDEN / f"{name}.txt"
    stored = text.replace("\x1b", "\\x1b")
    if os.environ.get(RECORD_VARIABLE) == "1":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stored.encode("ascii"))
        pytest.skip(f"recorded {path}; review it before committing")
    if not path.is_file():
        pytest.fail(f"no golden frame {path}: render it with {RECORD_VARIABLE}=1, review it, and commit it")
    assert stored == path.read_bytes().decode("ascii"), f"the frame differs from {path}"


@pytest.mark.parametrize("terminal", [True, False], ids=["terminal", "plain"])
@pytest.mark.parametrize("name", list(SCENARIOS), ids=list(SCENARIOS))
def test_each_scenario_matches_its_golden_frame(name: str, terminal: bool) -> None:
    text = render(SCENARIOS[name], terminal=terminal, width=100)
    _golden(f"{name}-{'terminal' if terminal else 'plain'}-w100", text)


# ---------------------------------------------------------------------------
# LiveTable, the observer that redraws the table


class FakeClock:
    """A clock the test sets: calling it returns `now` in seconds."""

    def __init__(self) -> None:
        """Start at 0."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the time the test set."""
        return self.now


def frames(stream: io.StringIO) -> int:
    """Return how many frames `stream` holds: one column row per frame."""
    return sum(1 for line in visible(stream.getvalue()).split("\n") if _COLUMN_ROW.match(line))


def last_frame(stream: io.StringIO) -> Frame:
    """Read back the last frame written to `stream`, with up to six header lines above its box."""
    lines = visible(stream.getvalue()).split("\n")
    column = max(index for index, line in enumerate(lines) if _COLUMN_ROW.match(line))
    start = column - 1
    while start > 0 and column - start <= 6 and not lines[start - 1].startswith(("|", "+")):
        start -= 1
    return parse_frame("\n".join(lines[start:]))


def story(number: int = 1, count: int = 1) -> list[tuple[str, Trial, str | None]]:
    """Return the events of one SYNTHETIC trial: a translation, a failed build, one correction, a clean build."""
    failed = attempt(0, "S1", broken=True, diagnostics=[COMPILE_ERROR])
    fixed = attempt(1, "S4")
    return [
        ("trial-start", trial([]), None),
        ("stage-start", trial([]), "generate"),
        ("request-sent", trial([]), "generate"),
        ("attempt", trial([attempt(0, "S1", broken=True)]), "generate"),
        ("stage-start", trial([attempt(0, "S1", broken=True)]), "compile_loop"),
        ("attempt", trial([failed]), "compile_loop"),
        ("request-sent", trial([failed]), "compile_loop"),
        ("attempt", trial([failed, attempt(1, "S1")]), "compile_loop"),
        ("attempt", trial([failed, fixed]), "compile_loop"),
        ("trial-end", trial([failed, fixed], Final(stage_reached="S4", corrections=1)), None),
    ]


def event(kind: str, case: Trial, stage: str | None, number: int = 2, count: int = 5) -> Any:
    """Return a ProgressEvent; a trial-start event carries the run id, the place, and the source."""
    module = progress()
    if kind == "trial-start":
        return module.ProgressEvent(kind=kind, trial=case, stage=stage, run_id=RUN, number=number, count=count,
                                    source=SOURCE)
    return module.ProgressEvent(kind=kind, trial=case, stage=stage)


def table(terminal: bool = True, width: int = 100) -> tuple[Any, io.StringIO, FakeClock]:
    """Return a LiveTable on a fresh stream with a fake clock, the stream, and the clock."""
    stream, clock = io.StringIO(), FakeClock()
    return live().LiveTable(stream, terminal=terminal, width=width, clock=clock), stream, clock


def test_the_first_event_draws_and_later_events_within_a_second_do_not() -> None:
    observer, stream, clock = table()
    steps = story()
    observer(event(*steps[0]))
    assert frames(stream) == 1, "the first event draws the table at once"
    for offset, step in enumerate(steps[1:-1], start=1):
        clock.now = 0.05 * offset
        observer(event(*step))
    assert frames(stream) == 1, "events within about a second of the last draw draw nothing"


def test_a_trial_end_always_draws_the_trials_final_frame() -> None:
    observer, stream, clock = table()
    for offset, step in enumerate(story()):
        clock.now = 0.05 * offset
        observer(event(*step))
    assert frames(stream) == 2
    rows = last_frame(stream).rows
    assert len(rows) == 1 + 2 + 1 and "final" in rows[-1].text(0).lower()


def test_an_event_about_a_second_and_a_half_after_the_last_draw_draws() -> None:
    observer, stream, clock = table()
    steps = story()
    for offset, step in enumerate(steps):
        clock.now = 1.5 * offset
        observer(event(*step))
    assert frames(stream) == len(steps)


def test_tick_redraws_about_once_a_second_with_the_elapsed_time() -> None:
    observer, stream, clock = table()
    observer(event(*story()[0]))
    clock.now = 0.5
    observer.tick()
    assert frames(stream) == 1, "a tick within about a second of the last draw draws nothing"
    clock.now = ELAPSED_S
    observer.tick()
    assert frames(stream) == 2
    assert ELAPSED_TEXT.search(" ".join(last_frame(stream).header)), "the elapsed time counts from the first event"
    clock.now = ELAPSED_S + 0.5
    observer.tick()
    assert frames(stream) == 2


def test_tick_before_any_event_draws_nothing() -> None:
    observer, stream, clock = table()
    clock.now = 5.0
    observer.tick()
    assert stream.getvalue() == ""


def test_the_header_comes_from_the_trial_start_event_and_the_trial() -> None:
    observer, stream, clock = table(width=200)
    observer(event(*story()[0], number=3, count=4))
    head = " ".join(last_frame(stream).header)
    for value in (RUN, ARM, DEVICE):
        assert value in head, f"{value!r} is not in the header: {head!r}"
    assert re.search(r"(?<![0-9])3\s*(?:/|of)\s*4(?![0-9])", head), head
    assert shown_code(last_frame(stream).rows[0], SOURCE_LINES) == excerpt(SOURCE_LINES, SOURCE_ANCHOR)


def test_while_a_request_is_out_the_frame_shows_the_latest_code_row() -> None:
    observer, stream, clock = table()
    steps = story()
    for offset, step in enumerate(steps[:7]):
        clock.now = 1.5 * offset
        observer(event(*step))
    working = last_frame(stream).rows
    assert len(working) == 1 + 1 + 1, "source, the failed translation, and the latest code while the model works"
    clock.now = 1.5 * 7
    observer(event(*steps[7]))
    assert len(last_frame(stream).rows) == 1 + 2, "the reply arrived: no latest code row"


def test_on_a_terminal_each_redraw_goes_back_over_the_last_frame() -> None:
    observer, stream, clock = table(terminal=True)
    observer(event(*story()[0]))
    first = stream.getvalue()
    clock.now = 2.0
    observer.tick()
    assert "\x1b[" in stream.getvalue()[len(first):], "a redraw on a terminal moves back over the last frame"


def test_without_a_terminal_frames_have_no_escape_and_no_code_column() -> None:
    observer, stream, clock = table(terminal=False)
    for offset, step in enumerate(story()):
        clock.now = 1.5 * offset
        observer(event(*step))
    assert "\x1b" not in stream.getvalue()
    assert frames(stream) >= 1
    shown = last_frame(stream)
    assert shown.columns == ["Step", "Status"]
    assert "final" in shown.rows[-1].text(0).lower(), "the trial's final frame is written"


# ---------------------------------------------------------------------------
# Fitting a frame to the terminal's height


def render_fitted(case: Trial, height: int, *, terminal: bool = True, width: int = 100) -> str:
    """Render `case` as a final frame fitted to `height` lines."""
    return live().render_table(case, header(), source=SOURCE, width=width, terminal=terminal, ended=True,
                               height=height)


def test_a_frame_taller_than_the_height_is_fitted_by_cutting_the_code_first() -> None:
    case = SCENARIOS["corrections"]
    full = render(case, width=100)
    text = render_fitted(case, 30)
    assert len(full.split("\n")) - 1 > 30 >= len(text.split("\n")) - 1
    frame = parse_frame(text)
    assert len(frame.rows) == len(parse_frame(full).rows), "every row stays while cutting the code is enough"
    shown = shown_code(frame.rows[0], SOURCE_LINES)
    expected = excerpt(SOURCE_LINES, SOURCE_ANCHOR)[: len(shown)]
    assert 0 < len(shown) < EXCERPT_LINES, shown
    assert all(cut_of(line, whole) for line, whole in zip(shown, expected, strict=True)), shown
    note = " ".join(frame.header)
    assert "30" in note and "code" in note, f"the header does not say the frame was fitted: {note!r}"


def test_when_cutting_the_code_is_not_enough_the_oldest_attempts_are_left_out() -> None:
    case = SCENARIOS["cap-hit"]
    text = render_fitted(case, 20)
    assert len(text.split("\n")) - 1 <= 20
    frame = parse_frame(text)
    steps = [row.text(0).lower() for row in frame.rows]
    assert "source" in steps[0] and "final" in steps[-1], steps
    assert "correction 3" in steps and "translation" not in steps, "the oldest attempts go first"
    assert all(not row.code() for row in frame.rows), "the code goes before any attempt row"
    assert "left out" in " ".join(frame.header)


def test_a_frame_that_cannot_fit_keeps_its_top_and_says_it_was_cut() -> None:
    text = render_fitted(SCENARIOS["corrections"], 5)
    lines = text.split("\n")[:-1]
    assert len(lines) == 5 and "cut" in lines[-1] and "height" in lines[-1], lines


def test_rendering_with_no_height_or_room_to_spare_is_the_plain_frame() -> None:
    case = SCENARIOS["clean-pass"]
    assert render_fitted(case, 500) == render(case, width=100)


def test_a_live_table_fits_its_frames_to_its_height_on_a_terminal() -> None:
    stream, clock = io.StringIO(), FakeClock()
    observer = live().LiveTable(stream, terminal=True, width=100, height=24, clock=clock)
    for offset, step in enumerate(story()):
        clock.now = 1.5 * offset
        observer(event(*step))
    frames_text = visible(stream.getvalue())
    assert frames(stream) == len(story())
    heights = [len(part.rstrip("\n").split("\n")) for part in frames_text.split("LASSI run ")[1:]]
    assert heights and max(heights) <= 24, heights


# ---------------------------------------------------------------------------
# The excerpt's anchor, and a table that raised


def test_the_anchor_is_a_whole_word_or_a_pragma_at_the_start_of_a_line() -> None:
    lines = [
        "int main() {",
        "  // #pragma omp in a comment is not a pragma line",
        "  int my__global__x = 0, kernel_mainly = 1;",
        "  #pragma omp parallel for",
        "  for (int i = 0; i < 4; ++i) { }",
        "}",
        "",
        "",
    ]
    name, first, shown = live().excerpt({"main.cpp": "\n".join(lines)})
    assert (name, first) == ("main.cpp", 4)
    assert shown == lines[3:6], "the excerpt runs from the pragma line and drops trailing blank lines"


class FailingStream(io.StringIO):
    """A stream whose first write raises OSError; it counts every write asked of it."""

    def __init__(self) -> None:
        """Start with no writes."""
        super().__init__()
        self.asked = 0

    def write(self, text: str) -> int:
        """Count the write; refuse the first."""
        self.asked += 1
        if self.asked == 1:
            raise OSError("SYNTHETIC stream failure")
        return super().write(text)


def test_a_table_that_raised_never_draws_again() -> None:
    stream, clock = FailingStream(), FakeClock()
    observer = live().LiveTable(stream, terminal=True, width=100, clock=clock)
    steps = story()
    with pytest.raises(OSError):
        observer(event(*steps[0]))
    for offset, step in enumerate(steps[1:], start=1):
        clock.now = 5.0 * offset
        observer.tick()
        observer(event(*step))
    assert stream.asked == 1 and stream.getvalue() == "", "a broken table was drawn again"


class DescriptorStream(FailingStream):
    """A FailingStream that reports a chosen file descriptor, as the process's stderr reports 2."""

    def __init__(self, descriptor: int) -> None:
        """Report `descriptor` from fileno()."""
        super().__init__()
        self.descriptor = descriptor

    def fileno(self) -> int:
        """Return the chosen descriptor."""
        return self.descriptor


def test_a_table_whose_write_to_stderr_fails_points_stderr_at_the_null_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = event(*story()[0])
    kept, muted = tmp_path / "kept.txt", tmp_path / "muted.txt"
    with kept.open("wb") as other, muted.open("wb") as target:
        stream = DescriptorStream(other.fileno())
        with pytest.raises(OSError):
            live().LiveTable(stream, terminal=True, width=100, clock=FakeClock())(first)
        stderr = DescriptorStream(target.fileno())
        monkeypatch.setattr(sys, "stderr", stderr)
        with pytest.raises(OSError):
            live().LiveTable(stderr, terminal=True, width=100, clock=FakeClock())(first)
        for opened in (other, target):
            os.write(opened.fileno(), b"SYNTHETIC bytes\n")
    assert kept.read_bytes() == b"SYNTHETIC bytes\n", "a stream other than sys.stderr was pointed elsewhere"
    assert muted.read_bytes() == b"", "a failed write to stderr left stderr's descriptor as it was"


def test_a_tick_that_raises_stops_the_ticks_with_one_notice(capsys: pytest.CaptureFixture[str]) -> None:
    module = live()

    class Ticks:
        """A table stand-in whose every tick raises and is counted."""

        def __init__(self) -> None:
            """Start uncounted."""
            self.ticks = 0

        def tick(self) -> None:
            """Count the tick and fail."""
            self.ticks += 1
            raise RuntimeError("SYNTHETIC tick failure")

    table = Ticks()
    with module.ticking(table, interval_s=0.01):  # type: ignore[arg-type]
        for _ in range(500):
            if table.ticks:
                break
            threading.Event().wait(0.01)
        threading.Event().wait(0.1)
    assert table.ticks == 1, "a tick that raised was followed by another"
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1 and "RuntimeError" in lines[0], lines
