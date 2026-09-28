"""Tests for `lassi run` with graphics on: the live inference table over a mock run (task P4.8).

Bible: Readability Standards, Terminal Presentation (the Inference table
bullet: redrawn in place about once a second; no Code column without an
interactive terminal; and the Rules: presentation writes no file, changes
no record, file, or exit status, and uses ANSI escape sequences only on a
terminal). plans/p4-ttsim.md, task P4.8 and the planning decision
"Progress hook (P4.8)".

The contract these tests fix, beside tests/present/test_live_table.py and
tests/present/test_progress_hook.py:

- With graphics on, `lassi run` passes the runner an observer
  (RunOptions.observer, a lassi.present.live.LiveTable) that draws the table
  on the command's terminal; with graphics off it passes none, so nothing
  changes. The table's width and height come from COLUMNS and LINES when
  they are set, else from the terminal stderr is on, else from
  shutil.get_terminal_size; its clock is lassi.present.live.clock, which the
  tests patch with a fake clock before the command runs. stdout is the same
  with graphics on as with graphics off, apart from wall times and the runs
  root, and a stderr that breaks under the table changes no file and no
  exit status, in the process too: a child `lassi --graphics on run`
  whose stderr pipe the parent closes before the banner, or after reading
  its first line, exits 0, as with graphics off, with the same stdout and
  run tree (the fix: lassi.core.progress.mute_stderr; without it Python's
  flush of stderr at exit fails and the process exits 120).
- On a fake terminal, over a mock run whose fake builds each take 0.1 s on
  the fake clock, the table is redrawn about once a second, never on every
  event: at most one frame per fake second plus two per trial (a trial's
  first and final frames) plus one, and at least each trial's final frame.
  A redraw on a terminal uses escape sequences; the last frame shows the
  last trial's final row and its place, 2 of 2; every box line fits
  COLUMNS.
- Without a terminal (`--graphics on` into pipes) no escape character is
  written and the table has no Code column and shows no code.
- `lassi run` with graphics on writes no file beyond what graphics off
  writes: the same run tree, apart from recorded times and dates, and no
  other file under the test directory, which holds HOME, TMPDIR, the
  working directory, and LASSI_CONFIG_DIR. A LiveTable that raises changes
  no file and no exit status.

The CLI runs with fake stdin, stdout, and stderr whose isatty() the test
chooses; frames are read from the ordered transcript of stdout and stderr
writes, so the tests do not fix which stream the table uses. The one
child-process test runs child_main (the same fakes, on the real clock) in
`python -c`, holding the command, then the run, until the parent writes a
line to stdin, so the banner, or else every table write, comes after the
pipe has lost its reader. `lassi run`
uses the real mock backend, generate and compile_loop stages, and none
executor, with a fake toolchain registered as nvcc-sm80 in place of the
default registry; the fake fails each trial's first build, so each of the
two trials takes one correction. git is faked so the commit cannot change
between two runs. The bench sources are small SYNTHETIC files. No value
here is a measurement.
"""

from __future__ import annotations

import copy
import importlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from lassi import cli
from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic
from lassi.core.registry import Registry
from lassi.core.stages import CompileLoopStage, GenerateStage
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend
from lassi.present.banner import BANNER

COLUMNS = 100
BUILD_S = 0.1
RUN_ID = "p48-live"
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
ESCAPE = "\x1b"

FAKE_OMP_SOURCE = "#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
FAKE_CUDA_SOURCE = "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"

# The p0-smoke recipe's values (tests/fixtures/recipes/p0-smoke.yaml) with two trials per item.
RECIPE_DATA: dict[str, Any] = {
    "extends": "base",
    "model": {"backend": "mock", "id": "mock-reference"},
    "llm": {"sampling": {"max_tokens": 4096}},
    "bench": {"suite": "lassi-hecbench-10", "split": "eval", "items": ["layout"]},
    "directions": [{"source": "omp", "target": "cuda"}],
    "prompts": "p0-smoke",
    "toolchain": {"cuda": "nvcc-sm80"},
    "stages": ["generate", "compile_loop"],
    "executor": {"kind": "none"},
    "trials": {"n": 2},
}

TIME_KEYS = frozenset({"wall_s", "date", "started_utc", "finished_utc"})
TIME_LABELS = frozenset({"wall_s", "Wall time (s)", "date", "Started (UTC)", "Finished (UTC)"})
ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")
_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OTHER_ESCAPES = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]")
_RULE = re.compile(r"^\+(?:[-=]+\+)+$")
_COLUMN_ROW = re.compile(r"^\|\s*Step\s*\|\s*Status\s*\|(?:\s*Code\s*\|)?$")


def live() -> ModuleType:
    """Import and return lassi.present.live."""
    return importlib.import_module("lassi.present.live")


# ---------------------------------------------------------------------------
# Environment, bench, recipe, and components


class FakeClock:
    """A clock that moves only when a fake build takes BUILD_S seconds."""

    def __init__(self) -> None:
        """Start at 0."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the fake time."""
        return self.now


@dataclass
class Scene:
    """One test's recipe, bench root, and fake clock."""

    recipe: Path
    bench: Path
    clock: FakeClock


@pytest.fixture
def scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Scene:
    """Put HOME, TMPDIR, the working directory, and the preset under tmp_path; register the fakes; write the inputs."""
    for name in ("LASSI_GRAPHICS", "LASSI_SCRATCH", "LASSI_RUNS_ROOT", "LASSI_TOOLCHAINS"):
        monkeypatch.delenv(name, raising=False)
    for name in ("home", "cwd", "compile-tmp", "config"):
        (tmp_path / name).mkdir()
    monkeypatch.setenv("LASSI_CONFIG_DIR", str(tmp_path / "config"))
    for name in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(name, str(tmp_path / "home"))
    for name in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(name, str(tmp_path / "compile-tmp"))
    monkeypatch.setenv("COLUMNS", str(COLUMNS))
    monkeypatch.setenv("LINES", "40")
    monkeypatch.chdir(tmp_path / "cwd")
    monkeypatch.setattr(runner_module, "_git", fake_git)
    settings = importlib.import_module("lassi.present.settings")
    monkeypatch.setattr(settings, "path_device", fake_path_device)
    clock = FakeClock()
    monkeypatch.setattr(runner_module, "DEFAULT_REGISTRY", fake_registry(clock))
    bench = tmp_path / "bench"
    for relative, text in (("src/layout-omp/main.cpp", FAKE_OMP_SOURCE), ("src/layout-cuda/main.cu", FAKE_CUDA_SOURCE)):
        path = bench / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    recipes = tmp_path / "recipes"
    recipes.mkdir()
    recipe = recipes / "p48-live.yaml"
    recipe.write_bytes(yaml.safe_dump(copy.deepcopy(RECIPE_DATA), sort_keys=False).encode("ascii"))
    return Scene(recipe=recipe, bench=bench, clock=clock)


def fake_git(*args: str) -> str:
    """Answer the runner's git calls with a fixed commit and a clean tree."""
    return {"rev-parse": f"{FAKE_COMMIT}\n", "status": ""}[args[0]]


def fake_path_device(path: Path) -> int:
    """Return device 1 for a filesystem root and 2 for any other path, so no test path is on a root filesystem."""
    return 1 if Path(path).parent == Path(path) else 2


def fake_registry(clock: FakeClock) -> Registry:
    """Return the registry `lassi run` uses here: the mock backend, the stages, the none executor, the fake nvcc."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    registry.register("Stage", "generate", GenerateStage)
    registry.register("Stage", "compile_loop", CompileLoopStage)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", timed_toolchain(clock))
    return registry


def timed_toolchain(clock: FakeClock) -> type:
    """Return a fake nvcc-sm80 whose every build takes BUILD_S on `clock` and fails in an attempt00 directory."""

    class TimedToolchain:
        """A Toolchain without PIN, built as factory(); it compiles nothing."""

        name = "nvcc-sm80"
        capabilities = frozenset({"diagnostics"})

        def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
            """Move the clock, write `files`, and fail the first attempt of each trial; else write an artifact."""
            clock.now += BUILD_S
            workdir = Path(workdir)
            for relative, text in files.items():
                target = workdir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            if "attempt00" in workdir.parts:
                error = Diagnostic(stage="compile", severity="error", code="fake-error", file="main.cu", line=1,
                                   column=1, message="SYNTHETIC compile error of the fake toolchain")
                return BuildResult(artifact=None, diagnostics=[error])
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return TimedToolchain


def run_argv(scene: Scene, runs_root: Path) -> list[str]:
    """Return the `lassi run` arguments for the two-trial recipe with the fake bench."""
    return ["run", str(scene.recipe), "--runs-root", str(runs_root), "--run-id", RUN_ID, "--bench-root",
            str(scene.bench)]


# ---------------------------------------------------------------------------
# Running the command line with fake streams


class FakeStream(io.StringIO):
    """A text stream with a chosen isatty() that logs each write, with its stream's label, in a shared list."""

    def __init__(self, label: str, writes: list[tuple[str, str]], tty: bool, text: str = "") -> None:
        """Start with `text` to read, reporting `tty` from isatty()."""
        super().__init__(text)
        self.label = label
        self.writes = writes
        self.tty = tty

    def isatty(self) -> bool:
        """Return the chosen answer."""
        return self.tty

    def write(self, text: str) -> int:
        """Log the write and keep the text."""
        self.writes.append((self.label, text))
        return super().write(text)


@dataclass
class Session:
    """One `lassi` invocation: its exit status, stdout, stderr, and every write in order."""

    code: int
    out: str
    err: str
    writes: list[tuple[str, str]] = field(default_factory=list)

    def transcript(self) -> str:
        """Return stdout and stderr as one terminal shows them: every write in order."""
        return "".join(text for label, text in self.writes if label in ("stdout", "stderr"))


def invoke(argv: Sequence[str], *, tty: bool) -> Session:
    """Run lassi.cli.main(argv) with fake streams, all terminals or none; fail the test on an argparse exit."""
    writes: list[tuple[str, str]] = []
    streams = (FakeStream("stdin", writes, tty), FakeStream("stdout", writes, tty), FakeStream("stderr", writes, tty))
    saved = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout, sys.stderr = streams
    try:
        code = cli.main(list(argv))
    except SystemExit as exited:
        pytest.fail(f"argparse exited ({exited.code}) on {list(argv)}: {streams[2].getvalue().strip()}")
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved
    return Session(code=code, out=streams[1].getvalue(), err=streams[2].getvalue(), writes=writes)


# ---------------------------------------------------------------------------
# Frames, read back from a transcript


def visible(text: str) -> str:
    """Return `text` without escape sequences and carriage returns: what a person reads."""
    return _OTHER_ESCAPES.sub("", _CSI.sub("", text)).replace("\r", "")


def column_rows(text: str) -> list[str]:
    """Return every column row (one per frame) in `text`, each stripped of escape sequences."""
    return [line for line in visible(text).split("\n") if _COLUMN_ROW.match(line)]


def box_lines(text: str) -> list[str]:
    """Return every rule and box line in `text`."""
    return [line for line in visible(text).split("\n") if _RULE.match(line) or line.startswith("|")]


def last_frame(text: str) -> tuple[list[str], list[list[str]]]:
    """Return the last frame's header lines (up to six above its box) and its data rows' joined cells."""
    lines = visible(text).split("\n")
    column = max(index for index, line in enumerate(lines) if _COLUMN_ROW.match(line))
    top = column - 1
    bounds = [position for position, char in enumerate(lines[top]) if char == "+"]
    head = [line for line in lines[max(0, top - 6):top] if not line.startswith(("|", "+"))]
    rows: list[list[str]] = []
    current: list[list[str]] | None = None
    for line in lines[column + 1:]:
        if _RULE.match(line):
            if current:
                rows.append([" ".join(" ".join(cell).split()) for cell in current])
            current = [[] for _ in bounds[1:]]
        elif line.startswith("|") and current is not None:
            for index in range(len(bounds) - 1):
                current[index].append(line[bounds[index] + 1:bounds[index + 1]])
        else:
            break
    return head, rows


# ---------------------------------------------------------------------------
# The run tree, apart from recorded times and dates


def _blank_times(value: Any) -> Any:
    """Return JSON data with the value of every TIME_KEYS key replaced by "<T>"."""
    if isinstance(value, dict):
        return {key: "<T>" if key in TIME_KEYS else _blank_times(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_blank_times(item) for item in value]
    return value


def _markdown(text: str) -> str:
    """Return Markdown with ISO times written <DATE> and each time cell of a table (TIME_LABELS) written <T>."""
    lines: list[str] = []
    header: list[str] | None = None
    for line in text.split("\n"):
        line = ISO_TIME.sub("<DATE>", line)
        if not line.startswith("|"):
            header = None
            lines.append(line)
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif not all(set(cell) <= set("-: ") for cell in cells):
            if len(cells) == 2 and cells[0] in TIME_LABELS:
                cells[1] = "<T>"
            cells = ["<T>" if index < len(header) and header[index] in TIME_LABELS else cell
                     for index, cell in enumerate(cells)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tree(run_dir: Path) -> dict[str, Any]:
    """Return every entry of a run tree by relative path: its content apart from recorded times and dates."""
    view: dict[str, Any] = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_dir():
            view[relative] = "<dir>"
        elif path.suffix == ".parquet":
            rows = pq.read_table(path).to_pylist()
            view[relative] = [{name: "<T>" if name.endswith(("wall_s", "_date")) else value
                               for name, value in row.items()} for row in rows]
        elif path.suffix == ".json":
            view[relative] = _blank_times(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix == ".md":
            view[relative] = _markdown(path.read_text(encoding="utf-8"))
        else:
            view[relative] = path.read_bytes()
    return view


def files_outside(root: Path, *run_roots: Path) -> list[str]:
    """Return every file under `root` outside `run_roots`, relative, sorted."""
    kept = [path for path in root.rglob("*") if path.is_file() and not any(r in path.parents for r in run_roots)]
    return sorted(path.relative_to(root).as_posix() for path in kept)


# ---------------------------------------------------------------------------
# Graphics on, on a terminal


def test_lassi_run_with_graphics_on_redraws_about_once_a_second_on_a_terminal(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live(), "clock", scene.clock)
    session = invoke(["--graphics", "on", *run_argv(scene, tmp_path / "runs-on")], tty=True)
    assert session.code == 0, session.err
    transcript = session.transcript()
    shown = column_rows(transcript)
    trials = 2
    most = int(scene.clock.now) + 2 * trials + 1
    assert trials <= len(shown) <= most, f"{len(shown)} frames over {scene.clock.now:.1f} fake seconds"
    assert all(re.search(r"\bCode\b", row) for row in shown), "a terminal shows the Code column"
    assert ESCAPE + "[" in transcript, "a redraw on a terminal uses escape sequences"
    too_wide = [line for line in box_lines(transcript) if len(line) > COLUMNS]
    assert not too_wide, f"box lines wider than COLUMNS={COLUMNS}: {too_wide[:2]}"
    head, rows = last_frame(transcript)
    assert re.search(r"(?<![0-9])2\s*(?:/|of)\s*2(?![0-9])", " ".join(head)), head
    final = f"{rows[-1][0]} {rows[-1][1]}"
    assert "final" in rows[-1][0].lower() and re.search(r"(?<![0-9.S])1\s+corrections?\b|\bcorrections?\W{0,3}1\b",
                                                        final, re.IGNORECASE), final


def test_lassi_run_with_graphics_on_writes_no_file_beyond_what_graphics_off_writes(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live(), "clock", scene.clock)
    off_root, on_root = tmp_path / "runs-off", tmp_path / "runs-on"
    off = invoke(["--graphics", "off", *run_argv(scene, off_root)], tty=True)
    assert off.code == 0, off.err
    before = files_outside(tmp_path, off_root, on_root)
    on = invoke(["--graphics", "on", *run_argv(scene, on_root)], tty=True)
    assert on.code == 0, on.err
    assert files_outside(tmp_path, off_root, on_root) == before, "graphics on wrote a file outside the run tree"
    assert sorted(path.name for path in on_root.iterdir()) == ["runs"]
    on_tree, off_tree = tree(on_root / "runs" / RUN_ID), tree(off_root / "runs" / RUN_ID)
    assert sorted(on_tree) == sorted(off_tree), "the run trees hold different files"
    assert [name for name in on_tree if on_tree[name] != off_tree[name]] == []


def test_a_live_table_that_raises_changes_no_file_and_no_exit_status(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = live()
    monkeypatch.setattr(module, "clock", scene.clock)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("SYNTHETIC table failure")

    monkeypatch.setattr(module.LiveTable, "__call__", fail)
    monkeypatch.setattr(module.LiveTable, "tick", fail)
    off_root, on_root = tmp_path / "runs-off", tmp_path / "runs-on"
    assert invoke(["--graphics", "off", *run_argv(scene, off_root)], tty=True).code == 0
    on = invoke(["--graphics", "on", *run_argv(scene, on_root)], tty=True)
    assert on.code == 0, on.err
    assert tree(on_root / "runs" / RUN_ID) == tree(off_root / "runs" / RUN_ID)


# ---------------------------------------------------------------------------
# Graphics on without a terminal, and graphics off


def test_lassi_run_with_graphics_on_without_a_terminal_writes_no_escape_and_no_code(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live(), "clock", scene.clock)
    session = invoke(["--graphics", "on", *run_argv(scene, tmp_path / "runs-on")], tty=False)
    assert session.code == 0, session.err
    assert ESCAPE not in session.out + session.err
    shown = column_rows(session.transcript())
    assert len(shown) >= 2, "each trial's final frame is written"
    assert all(not re.search(r"\bCode\b", row) for row in shown), "no Code column without a terminal"
    for fragment in ("__global__", "#pragma omp", "k<<<"):
        assert fragment not in session.out + session.err, f"code text {fragment!r} shown without a terminal"


@pytest.mark.parametrize(("option", "passed"), [("off", False), ("on", True)])
def test_only_graphics_on_passes_an_observer_to_the_run(
    option: str, passed: bool, scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live(), "clock", scene.clock)
    seen: list[Any] = []
    real = cli.run_recipe

    def spy(path: Path, options: Any) -> Path:
        seen.append(options)
        return real(path, options)

    monkeypatch.setattr(cli, "run_recipe", spy)
    session = invoke(["--graphics", option, *run_argv(scene, tmp_path / "runs")], tty=True)
    assert session.code == 0, session.err
    (options,) = seen
    assert (options.observer is not None) is passed


# ---------------------------------------------------------------------------
# stdout, a stderr that breaks, and the table's size


def _normalized(text: str, root: Path) -> str:
    """Return `lassi run` stdout with the runs root written <ROOT> and each wall time written <T>."""
    text = text.replace(str(root), "<ROOT>").replace(root.as_posix(), "<ROOT>")
    return re.sub(r"wall_s \S+", "wall_s <T>", text)


def test_stdout_with_graphics_on_is_stdout_with_graphics_off(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live(), "clock", scene.clock)
    off_root, on_root = tmp_path / "runs-off", tmp_path / "runs-on"
    off = invoke(["--graphics", "off", *run_argv(scene, off_root)], tty=True)
    on = invoke(["--graphics", "on", *run_argv(scene, on_root)], tty=True)
    assert off.code == 0 and on.code == 0, (off.err, on.err)
    assert off.out.strip(), "the plain run printed nothing to compare"
    assert _normalized(on.out, on_root) == _normalized(off.out, off_root)
    assert column_rows(on.err) and not column_rows(on.out), "the table goes to stderr, never stdout"


class BreaksAtTable(FakeStream):
    """A stderr that takes the banner, then raises BrokenPipeError from the table's first frame on."""

    broken = False

    def write(self, text: str) -> int:
        """Refuse every write from the first frame on."""
        if self.broken or "LASSI run" in text:
            self.broken = True
            raise BrokenPipeError(32, "SYNTHETIC broken pipe")
        return super().write(text)

    def flush(self) -> None:
        """Refuse the flush once broken."""
        if self.broken:
            raise BrokenPipeError(32, "SYNTHETIC broken pipe")


def test_a_stderr_that_breaks_under_the_table_changes_no_file_and_no_exit_status(
    scene: Scene, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(live(), "clock", scene.clock)
    off_root, on_root = tmp_path / "runs-off", tmp_path / "runs-on"
    assert invoke(["--graphics", "off", *run_argv(scene, off_root)], tty=True).code == 0
    writes: list[tuple[str, str]] = []
    saved = sys.stdin, sys.stdout, sys.stderr
    sys.stdin, sys.stdout = FakeStream("stdin", writes, True), FakeStream("stdout", writes, True)
    sys.stderr = stderr = BreaksAtTable("stderr", writes, True)
    try:
        code = cli.main(["--graphics", "on", *run_argv(scene, on_root)])
    finally:
        sys.stdin, sys.stdout, sys.stderr = saved
    assert stderr.broken, "the table never reached stderr"
    assert code == 0
    assert tree(on_root / "runs" / RUN_ID) == tree(off_root / "runs" / RUN_ID)


# The child process's program: import this module from its directory and run child_main on the arguments.
CHILD = (
    "import sys\n"
    "sys.path.insert(0, sys.argv[1])\n"
    "import test_run_live\n"
    "sys.exit(test_run_live.child_main(sys.argv[2:]))\n"
)


def child_main(argv: Sequence[str]) -> int:
    """In a child process: return `lassi` main()'s exit status for `argv`, with the scene fixture's fakes.

    The command starts once stdin gives a line and the run once it gives
    another, so the parent can close its end of the stderr pipe before the
    banner, or after the banner and before the table writes.
    """
    runner_module._git = fake_git
    importlib.import_module("lassi.present.settings").path_device = fake_path_device
    runner_module.DEFAULT_REGISTRY = fake_registry(FakeClock())
    start = cli.run_recipe

    def held(path: Path, options: Any) -> Path:
        """Wait for the parent's second line on stdin, then run the recipe."""
        sys.stdin.readline()
        return start(path, options)

    cli.run_recipe = held
    sys.stdin.readline()
    return cli.main(list(argv))


def run_child(scene: Scene, runs_root: Path, graphics: str, *, read_banner: bool) -> tuple[int, str, str]:
    """Run `lassi --graphics <graphics> run` in a child process whose stderr pipe loses its reader early.

    The parent closes its end of the pipe before the command starts, or,
    with `read_banner`, after reading the banner's first line and before
    the run starts. Returns the exit status, the stderr line read (empty
    without `read_banner`), and stdout.
    """
    read_end, write_end = os.pipe()
    argv = [sys.executable, "-c", CHILD, str(Path(__file__).parent), "--graphics", graphics,
            *run_argv(scene, runs_root)]
    try:
        child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=write_end)
    finally:
        os.close(write_end)
    with child, os.fdopen(read_end, "rb") as stderr:
        if not read_banner:
            stderr.close()
        assert child.stdin is not None
        child.stdin.write(b"start the command\n")
        child.stdin.flush()
        first = stderr.readline().decode("ascii") if read_banner else ""
        stderr.close()
        try:
            out, _ = child.communicate(b"start the run\n", timeout=120)
        except subprocess.TimeoutExpired:
            child.kill()
            raise
    return child.returncode, first, out.decode("ascii")


def test_a_stderr_pipe_whose_reader_has_gone_leaves_the_process_exit_status_0(
    scene: Scene, tmp_path: Path
) -> None:
    roots = {name: tmp_path / f"runs-{name}" for name in ("off", "after-banner", "before-banner")}
    off_code, _, off_out = run_child(scene, roots["off"], "off", read_banner=False)
    after_code, first, after_out = run_child(scene, roots["after-banner"], "on", read_banner=True)
    before_code, _, before_out = run_child(scene, roots["before-banner"], "on", read_banner=False)
    assert first.rstrip() == BANNER.splitlines()[0], "the banner did not reach the stderr pipe"
    assert (off_code, after_code, before_code) == (0, 0, 0), "a stderr pipe whose reader has gone changed the status"
    assert off_out.strip(), "the plain run printed nothing to compare"
    for name, out in (("after-banner", after_out), ("before-banner", before_out)):
        assert _normalized(out, roots[name]).splitlines() == _normalized(off_out, roots["off"]).splitlines(), name
        assert tree(roots[name] / "runs" / RUN_ID) == tree(roots["off"] / "runs" / RUN_ID), name


def closed_stream() -> io.StringIO:
    """Return a closed stream: every write raises ValueError."""
    stream = io.StringIO()
    stream.close()
    return stream


@pytest.mark.parametrize("stderr", [None, closed_stream()], ids=["none", "closed"])
def test_the_banner_passes_over_a_missing_or_closed_stderr(stderr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "stderr", stderr)
    cli._banner()


class TerminalStream(io.StringIO):
    """A stream on a pretend terminal, file descriptor 2."""

    def fileno(self) -> int:
        """Return 2."""
        return 2


def test_the_table_takes_its_size_from_stderrs_terminal_unless_columns_and_lines_are_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = {2: os.terminal_size((77, 33))}
    monkeypatch.setattr(cli.os, "get_terminal_size", lambda fd: sizes[fd])
    for name in ("COLUMNS", "LINES"):
        monkeypatch.delenv(name, raising=False)
    assert cli._table_size(TerminalStream()) == (77, 33)
    monkeypatch.setenv("COLUMNS", "120")
    assert cli._table_size(TerminalStream()) == (120, 33)
    monkeypatch.setenv("LINES", "50")
    assert cli._table_size(TerminalStream()) == (120, 50)
    monkeypatch.setenv("COLUMNS", "wide")
    assert cli._table_size(TerminalStream()) == (77, 50), "a COLUMNS that is not a number is passed over"
