"""The live inference table `lassi run` shows with graphics on (bible, Readability Standards, Terminal Presentation).

render_table draws one frame from a Trial snapshot and the header values,
as a pure function: no clock, no file, no terminal query. A frame is the
header lines, then a box in the style of `furiosa-smi status`: `+---+` rules
and `|` cells, a rule after the column row and after every row.

- Header: the run id and the trial's place in the run (`trial <n> of
  <count>`), the elapsed time as H:MM:SS, the arm and the trial's item and
  direction, and the device ("-" when the trial records none), wrapped to
  the width.
- Columns: Step, Status, and, on a terminal only, Code.
- Rows: the source program (Step `source`, the starter code); each attempt
  in order (Step `translation` for attempt 0, `correction <k>` for attempt
  k; Status its stage code and word, as `S1 parses`, then its first
  FIRST_ERRORS errors as `<stage> error: <message>` and how many more there
  are); while a model request is out (`working`), the latest code row; once
  the trial has ended (`ended`), the final product row (Status the final
  stage, the correction count, and the end reason's code and message when
  there is one).
- Code cells (terminal only): the source row shows the source files, and
  the latest code and final product rows the last attempt's files. A cell
  is one label line naming the file and the first line shown, then an
  excerpt of at most EXCERPT_LINES lines from the first line that ANCHOR
  finds, searched file by file in the files' order: `__global__`,
  `__kernel`, or `kernel_main` as a whole word (a kernel), `<<<` (a
  launch), or `#pragma` as the first text on the line (a pragma; blanks
  may come before it and between `#` and `pragma`); else from the first
  file's first line. Trailing blank lines are left out, so an excerpt
  can be shorter.
- Width: every line fits `width` (read as MIN_WIDTH when it is narrower);
  a code line longer than its cell is cut and ends with CUT, and other
  cells wrap.
- Height: with `height` given, a taller frame is fitted to it: the code
  excerpts are cut first, a line at a time down to none, then the oldest
  attempt rows are left out, and a header line says what was left out; a
  frame that still does not fit keeps its first `height` - 1 lines and a
  line saying it was cut.
- Text: every character outside printable ASCII in the source, a file
  name, a message, or a header value is written `?` (a tab first becomes
  spaces), so no control sequence from the code or a message reaches the
  output. render_table itself writes no escape sequence; messages are cut
  at MESSAGE_CHARS characters.

LiveTable is the progress observer (lassi.core.progress) that `lassi run`
passes with graphics on. It keeps the latest trial snapshot and draws
frames to its stream:

- on a terminal: at the first event it gets, at each trial-start, at each
  trial-end (the trial's final frame), and otherwise when REDRAW_S seconds
  have passed on its clock since its last draw, on an event or a tick().
  Frames are fitted to its height when it has one. Each frame after the
  first of a trial goes back over the last one with ANSI escape sequences
  (cursor up, erase to the end of the screen) and is drawn in its place; a
  trial's final frame stays, the next trial's frames start below it, and
  tick() draws nothing from a trial-end to the next trial-start;
- without a terminal: each trial's final frame only, whole, with no escape
  sequence and no Code column; tick() draws nothing.

The header's run id, place, and source come from the trial's trial-start
event, the arm from Trial.model.id, the device from
Trial.provenance.device, and the elapsed time counts from the first event.
From a request-sent event to the next attempt event the frame shows the
latest code row. A LiveTable that raises, in a call or a tick, is broken
for good: it passes the error on and never draws again, so the runner
drops it. When a write or flush to its stream fails with OSError (a pipe
whose reader has gone) and that stream is sys.stderr, it first points
stderr at the null device (lassi.core.progress.mute_stderr), so the bytes
left in stderr's buffer cannot fail the flush at interpreter exit and
change the process's exit status. ticking() calls tick() from a daemon
thread while a run goes on; a lock keeps a tick and an event from drawing
at once. Nothing here writes a file or reads the run tree.
"""

from __future__ import annotations

import re
import textwrap
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TextIO

from lassi.core.progress import (
    ATTEMPT,
    REQUEST_SENT,
    TRIAL_END,
    TRIAL_START,
    ProgressEvent,
    mute_stderr,
    stderr_notice,
)
from lassi.core.record import Attempt, Trial

clock: Callable[[], float] = time.monotonic
"""The table's monotonic clock in seconds; LiveTable reads it at call time when it is given none."""

REDRAW_S = 1.0
"""Seconds between redraws of a trial in progress on a terminal."""
TICK_S = 0.25
"""Seconds between the ticks ticking() sends, so a redraw comes within REDRAW_S + TICK_S of the last."""
EXCERPT_LINES = 12
MIN_WIDTH = 40
STEP_WIDTH = 13
FIRST_ERRORS = 2
MESSAGE_CHARS = 160
CUT = "~"
COLUMNS = ("Step", "Status", "Code")
ANCHOR = re.compile(r"\b__global__\b|\b__kernel\b|\bkernel_main\b|<<<|^\s*#\s*pragma\b")
STAGE_WORDS = {"S0": "no output", "S1": "parses", "S2": "verifies", "S3": "lowers", "S4": "compiles",
               "S5": "runs clean"}
FRAME_CUT = "(the frame is cut to the terminal's height)"

# The escape sequence that takes the cursor back to the start of a frame of `n` lines and erases to the end of
# the screen (CSI n A, cursor up; CSI J, erase below).
_BACK = "\r\x1b[{}A\x1b[J"

Cell = list[str]


@dataclass(frozen=True)
class TableHeader:
    """The header values of one frame: the run id, the arm, the device, the elapsed seconds, and the place."""

    run: str
    arm: str
    device: str | None
    elapsed_s: float
    number: int
    count: int


@dataclass(frozen=True)
class _View:
    """What one frame shows: render_table's arguments, with the width at least MIN_WIDTH."""

    trial: Trial
    header: TableHeader
    source: Mapping[str, str] | None
    width: int
    terminal: bool
    working: bool
    ended: bool


@dataclass(frozen=True)
class _Fit:
    """How a frame is fitted to a height: the excerpt lines each code cell keeps and the oldest attempts left out."""

    code_lines: int = EXCERPT_LINES
    hidden: int = 0


# ---------------------------------------------------------------------------
# One frame


def render_table(
    trial: Trial,
    header: TableHeader,
    *,
    source: Mapping[str, str] | None,
    width: int,
    terminal: bool,
    working: bool = False,
    ended: bool = False,
    height: int | None = None,
) -> str:
    """Return one frame of the table for `trial`, ending in a line feed; a pure function of its arguments.

    `source` is the starter code (file name -> text), `working` says a
    model request is out, and `ended` that the trial has ended. `terminal`
    adds the Code column. With `height` (read as at least 1) the frame is
    fitted to that many lines. The module docstring gives the layout.
    """
    view = _View(trial, header, source, max(width, MIN_WIDTH), terminal, working, ended)
    lines = _frame(view, _Fit(), None)
    if height is not None and len(lines) > max(height, 1):
        lines = _fitted(view, max(height, 1))
    return "\n".join(lines) + "\n"


def _frame(view: _View, fit: _Fit, height: int | None) -> list[str]:
    """Return a frame's lines as `fit` cuts it; `height` is named in the header's note when anything is cut."""
    widths = _widths(view.width, view.terminal)
    rows = [[[name] for name in COLUMNS[: len(widths)]]]
    for step, status, files in _rows(view, fit.hidden):
        cells = [_wrapped(step, widths[0]), _wrapped(status, widths[1])]
        if view.terminal:
            cells.append(_code_cell(files, widths[2], fit.code_lines))
        rows.append(cells)
    head = _header(view.trial, view.header, view.width)
    if fit != _Fit():
        head += _wrapped(_fit_note(fit, height), view.width)
    return head + _box(widths, rows)


def _fitted(view: _View, height: int) -> list[str]:
    """Return the frame fitted to `height` lines: excerpts cut first, then the oldest attempts left out."""
    kept = 0 if view.terminal else EXCERPT_LINES
    lines = _frame(view, _Fit(kept, 0), height)
    if view.terminal:
        for code_lines in range(EXCERPT_LINES - 1, -1, -1):
            lines = _frame(view, _Fit(code_lines, 0), height)
            if len(lines) <= height:
                return lines
    for hidden in range(1, len(view.trial.attempts) + 1):
        lines = _frame(view, _Fit(kept, hidden), height)
        if len(lines) <= height:
            return lines
    return lines[: height - 1] + [_cut(FRAME_CUT, view.width)]


def _fit_note(fit: _Fit, height: int | None) -> str:
    """Return the header line that says what fitting the frame to `height` lines left out."""
    parts = []
    if fit.code_lines < EXCERPT_LINES:
        parts.append(f"code cut to {fit.code_lines} line(s) a cell" if fit.code_lines else "code left out")
    if fit.hidden:
        parts.append("attempt 0 left out" if fit.hidden == 1 else f"attempts 0 to {fit.hidden - 1} left out")
    return f"(to fit {height} terminal lines: {'; '.join(parts)})"


def _widths(width: int, terminal: bool) -> list[int]:
    """Return the column widths for a frame `width` wide: Step, Status, and, on a terminal, Code."""
    count = 3 if terminal else 2
    room = width - (3 * count + 1)
    step = min(STEP_WIDTH, max(8, room // 4))
    rest = room - step
    if not terminal:
        return [step, rest]
    status = max(8, rest * 2 // 5)
    return [step, status, rest - status]


def _rows(view: _View, hidden: int) -> list[tuple[str, str, Mapping[str, str] | None]]:
    """Return each row's Step text, Status text, and the files its Code cell shows (None for no code).

    The first `hidden` attempts get no row.
    """
    trial = view.trial
    rows: list[tuple[str, str, Mapping[str, str] | None]] = [("source", "the starter code", view.source)]
    for attempt in trial.attempts[hidden:]:
        step = "translation" if attempt.index == 0 else f"correction {attempt.index}"
        rows.append((step, _attempt_status(attempt), None))
    last = trial.attempts[-1].files if trial.attempts else None
    if view.working:
        rows.append(("latest code", "waiting for the model", last))
    if view.ended:
        rows.append(("final product", _final_status(trial), last))
    return rows


def _stage(code: str | None) -> str:
    """Return a stage code with its word, as `S4 compiles`; `no stage` for None."""
    if code is None:
        return "no stage"
    word = STAGE_WORDS.get(code)
    return code if word is None else f"{code} {word}"


def _attempt_status(attempt: Attempt) -> str:
    """Return an attempt's Status text: its stage, then its first errors and how many more there are."""
    parts = [_stage(attempt.stage_reached)]
    errors = [item for item in attempt.diagnostics if item.severity == "error"]
    parts += [f"{item.stage} error: {_message(item.message)}" for item in errors[:FIRST_ERRORS]]
    if len(errors) > FIRST_ERRORS:
        parts.append(f"{len(errors) - FIRST_ERRORS} more error(s)")
    return "; ".join(parts)


def _final_status(trial: Trial) -> str:
    """Return the final row's Status text: the final stage, the correction count, and the end reason."""
    final = trial.final
    count = final.corrections
    parts = [_stage(final.stage_reached), f"{count} correction{'' if count == 1 else 's'}"]
    if final.end_reason is not None:
        parts.append(f"end {final.end_reason.code}: {_message(final.end_reason.message)}")
    return "; ".join(parts)


def _plain(text: str) -> str:
    """Return `text` with tabs as spaces and every other character outside printable ASCII written `?`."""
    return "".join(char if " " <= char <= "~" else "?" for char in text.expandtabs(4))


def _message(text: str) -> str:
    """Return a message on one line of plain ASCII, cut at MESSAGE_CHARS characters."""
    text = _plain(" ".join(text.split()))
    return text if len(text) <= MESSAGE_CHARS else text[: MESSAGE_CHARS - 3] + "..."


def _cut(text: str, width: int) -> str:
    """Return `text` cut to `width` characters, ending in CUT when it was longer."""
    return text if len(text) <= width else text[: width - len(CUT)] + CUT


def _wrapped(text: str, width: int) -> Cell:
    """Return `text` in plain ASCII wrapped to `width`, one list item per line."""
    return textwrap.wrap(_plain(text), width, break_on_hyphens=False) or [""]


def excerpt(files: Mapping[str, str]) -> tuple[str, int, list[str]] | None:
    """Return the code excerpt of `files`: the file name, its first line's number, and the lines as written.

    The excerpt is at most EXCERPT_LINES lines from the first line that
    ANCHOR finds, searched file by file in the mapping's order, else the
    first file's first lines, with trailing blank lines left out. CRLF
    counts as one line break. None when there are no files.
    """
    if not files:
        return None
    split = {name: text.replace("\r\n", "\n").split("\n") for name, text in files.items()}
    for name, lines in split.items():
        for index, line in enumerate(lines):
            if ANCHOR.search(line):
                return name, index + 1, _trimmed(lines[index : index + EXCERPT_LINES])
    name, lines = next(iter(split.items()))
    return name, 1, _trimmed(lines[:EXCERPT_LINES])


def _trimmed(lines: Sequence[str]) -> list[str]:
    """Return `lines` without its trailing blank lines."""
    kept = list(lines)
    while kept and not kept[-1].strip():
        kept.pop()
    return kept


def _code_cell(files: Mapping[str, str] | None, width: int, code_lines: int) -> Cell:
    """Return a Code cell: a label line naming the file and first line, then `code_lines` excerpt lines at most.

    Each line is cut to `width`; with `code_lines` 0 the cell is empty.
    """
    found = excerpt(files or {})
    if found is None or code_lines <= 0:
        return [""]
    name, first, lines = found
    label = _cut(_plain(f"{name}, from line {first}"), width)
    return [label] + [_cut(_plain(line), width) for line in _trimmed(lines[:code_lines])]


def _header(trial: Trial, header: TableHeader, width: int) -> list[str]:
    """Return the header lines: the run, place, and elapsed time; the arm and item; the device."""
    item = trial.bench_item
    fields = (
        f"LASSI run {header.run}  trial {header.number} of {header.count}  elapsed {_elapsed(header.elapsed_s)}",
        f"arm {header.arm}  item {item.item} ({item.direction})",
        f"device {header.device if header.device is not None else '-'}",
    )
    lines: list[str] = []
    for text in fields:
        lines += _wrapped(text, width)
    return lines


def _elapsed(seconds: float) -> str:
    """Return `seconds` as H:MM:SS, whole seconds, never below 0:00:00."""
    whole = int(max(0.0, seconds))
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def _box(widths: Sequence[int], rows: Sequence[Sequence[Cell]]) -> list[str]:
    """Return the box lines: a rule, then each row's lines (cells padded to their widths) and a rule after it."""
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [rule]
    for cells in rows:
        for index in range(max(len(cell) for cell in cells)):
            texts = [cell[index] if index < len(cell) else "" for cell in cells]
            lines.append("|" + "|".join(f" {text:<{width}} " for text, width in zip(texts, widths, strict=True)) + "|")
        lines.append(rule)
    return lines


# ---------------------------------------------------------------------------
# The observer that redraws the table


class LiveTable:
    """The progress observer that draws the table to a stream; the module docstring says when it draws."""

    def __init__(
        self,
        stream: TextIO,
        *,
        terminal: bool,
        width: int,
        height: int | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Draw to `stream`, `width` columns wide; `terminal` says it is an interactive terminal.

        On a terminal each frame is fitted to `height` lines when it is given.
        `clock` gives seconds; None means the module's `clock`, read at each call.
        """
        self._stream = stream
        self._terminal = terminal
        self._width = width
        self._height = height
        self._clock = clock
        self._lock = threading.Lock()
        self._trial: Trial | None = None
        self._run, self._number, self._count = "", 0, 0
        self._source: Mapping[str, str] | None = None
        self._first: float | None = None
        self._last: float | None = None
        self._lines = 0
        self._working = self._ended = self._broken = False

    def __call__(self, event: ProgressEvent) -> None:
        """Keep the event's trial and draw a frame when one is due; any error breaks the table for good."""
        with self._lock:
            if self._broken:
                return
            try:
                self._take_and_draw(event)
            except BaseException:
                self._broken = True
                raise

    def tick(self) -> None:
        """Redraw the trial in progress on a terminal when REDRAW_S seconds have passed since the last draw.

        Any error breaks the table for good, as in a call.
        """
        with self._lock:
            if self._broken or not self._terminal or self._trial is None or self._ended:
                return
            try:
                now = self._now()
                if self._due(now):
                    self._draw(now)
            except BaseException:
                self._broken = True
                raise

    def _take_and_draw(self, event: ProgressEvent) -> None:
        """Keep what the event says and draw when the module docstring says a frame is due."""
        now = self._now()
        first = self._first is None
        if first:
            self._first = now
        self._take(event)
        if not self._terminal:
            if event.kind == TRIAL_END:
                self._draw(now)
            return
        if first or event.kind in (TRIAL_START, TRIAL_END) or self._due(now):
            self._draw(now)

    def _now(self) -> float:
        """Return the time on the table's clock, or on the module's clock when it was given none."""
        return (clock if self._clock is None else self._clock)()

    def _due(self, now: float) -> bool:
        """Return True when REDRAW_S seconds have passed since the last draw, or nothing was drawn yet."""
        return self._last is None or now - self._last >= REDRAW_S

    def _take(self, event: ProgressEvent) -> None:
        """Keep what the event says: the trial, the header values of a trial-start, and whether the model works."""
        if event.kind == TRIAL_START:
            self._run = event.run_id or ""
            self._number = event.number or 0
            self._count = event.count or 0
            self._source = event.source
            self._ended = False
        self._trial = event.trial
        if event.kind == REQUEST_SENT:
            self._working = True
        elif event.kind in (ATTEMPT, TRIAL_START, TRIAL_END):
            self._working = False
        if event.kind == TRIAL_END:
            self._ended = True

    def _draw(self, now: float) -> None:
        """Write the frame for `now`, over the trial's last frame on a terminal, fitted to the height there.

        A write or flush that raises OSError calls mute_stderr on the stream
        (it acts only on sys.stderr) and passes the error on.
        """
        trial = self._trial
        if trial is None:
            return
        header = TableHeader(
            run=self._run, arm=trial.model.id, device=trial.provenance.device,
            elapsed_s=now - (now if self._first is None else self._first), number=self._number, count=self._count,
        )
        frame = render_table(
            trial, header, source=self._source, width=self._width, terminal=self._terminal, working=self._working,
            ended=self._ended, height=self._height if self._terminal else None,
        )
        back = _BACK.format(self._lines) if self._terminal and self._lines else ""
        try:
            self._stream.write(back + frame)
            self._stream.flush()
        except OSError:
            mute_stderr(self._stream)
            raise
        self._last = now
        self._lines = 0 if self._ended else frame.count("\n")


@contextmanager
def ticking(table: LiveTable, interval_s: float = TICK_S) -> Iterator[LiveTable]:
    """Call table.tick() from a daemon thread every `interval_s` seconds while the block runs.

    A tick that raises stops the ticks for good, as the runner drops an
    observer that raises, with one line on stderr naming the exception's
    class when stderr can take it; nothing reaches the block. The thread is
    stopped and joined (for at most a few seconds) when the block ends.
    """
    stop = threading.Event()

    def loop() -> None:
        """Tick until stopped or until a tick raises."""
        while not stop.wait(interval_s):
            try:
                table.tick()
            except Exception as error:  # a table that raises gets no more ticks
                stderr_notice(f"lassi: the live table raised {type(error).__name__}; it is not redrawn again")
                return

    thread = threading.Thread(target=loop, name="lassi-live-table", daemon=True)
    thread.start()
    try:
        yield table
    finally:
        stop.set()
        thread.join(timeout=5.0)
