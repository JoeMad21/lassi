"""Tests for the progress hook the runner and stages send to an in-process observer (task P4.8).

Bible: Readability Standards, Terminal Presentation (the Delivery bullet's
[OPEN] on how the runner hands live progress to the table, and the Rules:
presentation changes no stage, score, record, file, or exit status);
Component Interfaces, Stage contract rules (stages stay pure over the trial
record); Design Principle 2 (one result record). plans/p4-ttsim.md, the
planning decision "Progress hook (P4.8)".

The contract these tests fix:

- lassi.core.progress holds the hook's vocabulary. ProgressEvent is a frozen
  dataclass with `kind`, `trial` (the immutable Trial at that moment),
  `stage` (the registered stage name, set on stage-start and request-sent
  events), and four values the runner sets on each trial-start event:
  `run_id` (the run directory's name), `number` (the trial's 1-based place
  in the run), `count` (the run's trial count), and `source` (the item's
  files in the direction's source language, file name -> text, as
  Suite.source_files reads them), which the table's header and source row
  need; `stage` and those four default to None. The kinds are TRIAL_START "trial-start", STAGE_START "stage-start",
  REQUEST_SENT "request-sent", ATTEMPT "attempt" (an attempt appended or
  updated), and TRIAL_END "trial-end", listed in that order in EVENT_KINDS;
  any other kind is a ValueError.
- RunOptions.observer and RunContext.observer, both None by default: the
  observer is any callable taking one ProgressEvent. The runner sends
  trial-start, stage-start (before each stage it runs), and trial-end (the
  trial as written); the stages send request-sent (before the reply is
  recorded, so the k-th request-sent event's trial holds k requests) and
  attempt events, through the RunContext.
- The observer writes nothing and nothing depends on it: a mock run writes
  the same run tree, apart from recorded times and dates, and run_recipe
  returns normally, with no observer, a recording one, and one that raises.
  An observer that raises is dropped for the rest of the run: it is never
  called again, in that trial or any later one. It leaves one line on
  stderr naming the exception's class, when stderr can take it; a stderr
  that cannot (a broken pipe, a closed stream, or none) changes nothing
  else.
- A notice whose write or flush raises OSError calls mute_stderr, which
  points the file descriptor under sys.stderr at the null device (os.dup2);
  a stream that is not sys.stderr is left alone, and a failed dup2 raises
  nothing. The test's stderr stand-ins report descriptors of the test's
  own files, so the test process's stderr is never touched.

The runs use the real mock backend, the generate, compile_loop, and
baseline stages, and the none executor, with a fake toolchain registered as
nvcc-sm80 in a test Registry (as tests/core/test_runner.py does); the fake
fails a build whose workdir is an attempt00 directory when asked to, so a
trial takes one correction, and a build of a file with an `#error` line.
git is faked as tests/core/test_runner.py fakes it, so the commit cannot
change between two runs of one test. The bench sources are small
SYNTHETIC files. No value here is a measurement.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import io
import json
import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from lassi.core import runner as runner_module
from lassi.core.interfaces import BuildResult
from lassi.core.record import Diagnostic, Trial
from lassi.core.registry import Registry
from lassi.core.runner import RunOptions, run_recipe
from lassi.core.stages import BaselineStage, CompileLoopStage, GenerateStage, RunContext
from lassi.core.store import TextStore, read_trial, trial_dir
from lassi.executors import NoneExecutor
from lassi.llm import MockBackend

SUITE = "lassi-hecbench-10"
ITEM = "layout"
RUN_ID = "p48-hook"
# A commit id the faked git reports; not a commit of this repository.
FAKE_COMMIT = "0123456789abcdef0123456789abcdef01234567"
KINDS = ("trial-start", "stage-start", "request-sent", "attempt", "trial-end")

# SYNTHETIC bench sources standing in for the layout item; the mock backend replies with the CUDA one.
FAKE_OMP_SOURCE = "#include <cstdio>\nint main() {\n#pragma omp target\n  { }\n  return 0;\n}\n"
FAKE_CUDA_SOURCE = "__global__ void k() { }\nint main() {\n  k<<<1, 1>>>();\n  return 0;\n}\n"
BROKEN_CUDA_SOURCE = "#error SYNTHETIC broken reference\nint main() { return 0; }\n"

# The p0-smoke recipe's values (tests/fixtures/recipes/p0-smoke.yaml) with two trials per item.
RECIPE_DATA: dict[str, Any] = {
    "extends": "base",
    "model": {"backend": "mock", "id": "mock-reference"},
    "llm": {"sampling": {"max_tokens": 4096}},
    "bench": {"suite": SUITE, "split": "eval", "items": [ITEM]},
    "directions": [{"source": "omp", "target": "cuda"}],
    "prompts": "p0-smoke",
    "toolchain": {"cuda": "nvcc-sm80"},
    "stages": ["generate", "compile_loop"],
    "executor": {"kind": "none"},
    "trials": {"n": 2},
}

# Keys of trial.json and provenance.json, and labels of run.md and trial.md tables, whose values are times or dates.
TIME_KEYS = frozenset({"wall_s", "date", "started_utc", "finished_utc"})
TIME_LABELS = frozenset({"wall_s", "Wall time (s)", "date", "Started (UTC)", "Finished (UTC)"})
ISO_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")


# ---------------------------------------------------------------------------
# Modules under test, imported per test so each test fails on its own


def progress() -> ModuleType:
    """Import and return lassi.core.progress."""
    return importlib.import_module("lassi.core.progress")


# ---------------------------------------------------------------------------
# Environment, bench, recipe, and components


@pytest.fixture(autouse=True)
def environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset the gate variables, put TMPDIR under the test directory, and fake git's answers."""
    for name in ("LASSI_RUNS_ROOT", "LASSI_SCRATCH", "LASSI_TOOLCHAINS", "LASSI_GRAPHICS"):
        monkeypatch.delenv(name, raising=False)
    tmpdir = tmp_path / "compile-tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    answers = {"rev-parse": f"{FAKE_COMMIT}\n", "status": ""}
    monkeypatch.setattr(runner_module, "_git", lambda *args: answers[args[0]])


def write_bench(root: Path, cuda_source: str = FAKE_CUDA_SOURCE) -> Path:
    """Write the SYNTHETIC layout sources under `root` as the suite manifest lays them out; return `root`."""
    for relative, text in (("src/layout-omp/main.cpp", FAKE_OMP_SOURCE), ("src/layout-cuda/main.cu", cuda_source)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("ascii"))
    return root


def write_recipe(directory: Path, name: str, **changes: Any) -> Path:
    """Write RECIPE_DATA with top-level `changes` as `<directory>/<name>.yaml` and return its path."""
    data = copy.deepcopy(RECIPE_DATA)
    data.update(changes)
    path = directory / f"{name}.yaml"
    path.write_bytes(yaml.safe_dump(data, sort_keys=False).encode("ascii"))
    return path


def fake_toolchain(*, fail_first: bool = False, fail_always: bool = False) -> type:
    """Return a Toolchain class without PIN, registered as nvcc-sm80, that writes the files it builds.

    A build fails, with one compile error, when `fail_always` is set, when
    `fail_first` is set and its workdir lies in an attempt00 directory (the
    first attempt of each trial), or when a file holds an `#error` line;
    any other build writes a placeholder artifact. It runs no command.
    """

    class FakeToolchain:
        """A Toolchain without PIN, built as factory(); it compiles nothing."""

        name = "nvcc-sm80"
        capabilities = frozenset({"diagnostics"})

        def build(self, files: Mapping[str, str], workdir: Path) -> BuildResult:
            """Write `files` under `workdir`; return an error or a placeholder artifact as the class docstring says."""
            workdir = Path(workdir)
            for relative, text in files.items():
                target = workdir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(text.encode("utf-8"))
            first = "attempt00" in workdir.parts
            marked = any(line.startswith("#error") for text in files.values() for line in text.split("\n"))
            if fail_always or (fail_first and first) or marked:
                error = Diagnostic(
                    stage="compile", severity="error", code="fake-error", file="main.cu", line=1, column=1,
                    message="SYNTHETIC compile error of the fake toolchain",
                )
                return BuildResult(artifact=None, diagnostics=[error])
            artifact = workdir / "main"
            artifact.write_bytes(b"PLACEHOLDER artifact of the fake toolchain\n")
            return BuildResult(artifact=artifact, diagnostics=[])

    return FakeToolchain


def make_registry(toolchain: type, stage_classes: Mapping[str, type] | None = None) -> Registry:
    """Return a test Registry: the mock backend, the real stages (or `stage_classes`), none, and `toolchain`."""
    registry = Registry()
    registry.register("LLMBackend", "mock", MockBackend)
    chosen = {"baseline": BaselineStage, "generate": GenerateStage, "compile_loop": CompileLoopStage}
    chosen.update(stage_classes or {})
    for name, stage_class in chosen.items():
        registry.register("Stage", name, stage_class)
    registry.register("Executor", "none", NoneExecutor)
    registry.register("Toolchain", "nvcc-sm80", toolchain)
    return registry


_NO_OBSERVER = object()


def run(recipe: Path, runs_root: Path, bench: Path, registry: Registry, observer: Any = _NO_OBSERVER) -> Path:
    """Run `recipe` under `runs_root` with run id RUN_ID; pass `observer` only when one is given."""
    options: dict[str, Any] = {"runs_root": runs_root, "run_id": RUN_ID, "bench_root": bench, "registry": registry}
    if observer is not _NO_OBSERVER:
        options["observer"] = observer
    return run_recipe(recipe, RunOptions(**options))


# ---------------------------------------------------------------------------
# Observers


class Recorder:
    """An observer that keeps every event it is given."""

    def __init__(self) -> None:
        """Start with no events."""
        self.events: list[Any] = []

    def __call__(self, event: Any) -> None:
        """Keep the event."""
        self.events.append(event)


class Raiser:
    """An observer that raises RuntimeError at the first event `when` accepts, and counts every call."""

    def __init__(self, when: Callable[[Any], bool]) -> None:
        """Keep the test that picks the event to raise at."""
        self.when = when
        self.calls = 0
        self.raised_at: int | None = None

    def __call__(self, event: Any) -> None:
        """Count the call; raise when `when` accepts the event (every call after that also counts)."""
        self.calls += 1
        if self.raised_at is None and self.when(event):
            self.raised_at = self.calls
            raise RuntimeError("SYNTHETIC observer failure")


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


def _parquet(path: Path) -> list[dict[str, Any]]:
    """Return a Parquet file's rows with every wall time and date column written <T>."""
    rows = pq.read_table(path).to_pylist()
    timed = [name for name in (rows[0] if rows else {}) if name.endswith("wall_s") or name.endswith("_date")]
    return [{name: "<T>" if name in timed else value for name, value in row.items()} for row in rows]


def tree(run_dir: Path) -> dict[str, Any]:
    """Return every entry of a run tree by relative path: its content apart from recorded times and dates."""
    view: dict[str, Any] = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_dir():
            view[relative] = "<dir>"
        elif path.suffix == ".parquet":
            view[relative] = _parquet(path)
        elif path.suffix == ".json":
            view[relative] = _blank_times(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix == ".md":
            view[relative] = _markdown(path.read_text(encoding="utf-8"))
        else:
            view[relative] = path.read_bytes()
    return view


def assert_same_tree(actual: Path, expected: Path) -> None:
    """Fail unless two run trees hold the same entries with the same content apart from times and dates."""
    left, right = tree(actual), tree(expected)
    assert sorted(left) == sorted(right), "the run trees hold different files"
    different = [name for name in left if left[name] != right[name]]
    assert not different, f"these files differ beyond recorded times and dates: {different}"


# ---------------------------------------------------------------------------
# Scenarios


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Return a bench root holding the SYNTHETIC layout sources."""
    return write_bench(tmp_path / "bench")


def trials_of(events: list[Any]) -> list[list[Any]]:
    """Split events into one list per trial, each from its trial-start event to its trial-end event."""
    grouped: list[list[Any]] = []
    for event in events:
        if event.kind == "trial-start":
            grouped.append([])
        assert grouped, f"an event came before any trial-start: {event.kind}"
        grouped[-1].append(event)
    return grouped


def written_trial(run_dir: Path, trial_id: str) -> Trial:
    """Read one trial back from the run tree."""
    return read_trial(trial_dir(run_dir, trial_id), TextStore(run_dir))


# ---------------------------------------------------------------------------
# The vocabulary


def test_the_event_kinds_are_the_planning_decisions_five_in_order() -> None:
    module = progress()
    assert module.EVENT_KINDS == KINDS
    names = ("TRIAL_START", "STAGE_START", "REQUEST_SENT", "ATTEMPT", "TRIAL_END")
    assert tuple(getattr(module, name) for name in names) == KINDS


def test_a_progress_event_is_frozen_and_refuses_an_unknown_kind(tmp_path: Path, bench: Path) -> None:
    module = progress()
    recorder = Recorder()
    run(write_recipe(tmp_path, "p48-hook", trials={"n": 1}), tmp_path / "runs", bench,
        make_registry(fake_toolchain()), recorder)
    event = recorder.events[0]
    assert isinstance(event, module.ProgressEvent)
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.kind = "trial-end"  # type: ignore[misc]
    with pytest.raises(ValueError):
        module.ProgressEvent(kind="trial-paused", trial=event.trial)


def test_run_options_and_run_context_take_an_observer_that_defaults_to_none() -> None:
    for cls in (RunOptions, RunContext):
        fields = {field.name: field for field in dataclasses.fields(cls)}
        assert "observer" in fields, f"{cls.__name__} has no observer field"
        assert fields["observer"].default is None, f"{cls.__name__}.observer does not default to None"


# ---------------------------------------------------------------------------
# Nothing depends on the observer


OBSERVERS: dict[str, Callable[[], Any]] = {
    "recording": Recorder,
    "raising-at-once": lambda: Raiser(lambda event: True),
    "raising-at-an-attempt": lambda: Raiser(lambda event: event.kind == "attempt"),
    "raising-at-a-request": lambda: Raiser(lambda event: event.kind == "request-sent"),
    "raising-at-trial-end": lambda: Raiser(lambda event: event.kind == "trial-end"),
}


@pytest.mark.parametrize("fail_first", [False, True], ids=["clean", "one-correction"])
@pytest.mark.parametrize("name", list(OBSERVERS), ids=list(OBSERVERS))
def test_a_mock_run_writes_the_same_tree_with_or_without_an_observer(
    name: str, fail_first: bool, tmp_path: Path, bench: Path
) -> None:
    recipe = write_recipe(tmp_path, "p48-hook")
    registry = make_registry(fake_toolchain(fail_first=fail_first))
    plain = run(recipe, tmp_path / "plain", bench, registry)
    observer = OBSERVERS[name]()
    observed = run(recipe, tmp_path / "observed", bench, registry, observer)
    assert observed == tmp_path / "observed" / "runs" / RUN_ID
    assert_same_tree(observed, plain)
    manifest = json.loads((observed / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"


@pytest.mark.parametrize(
    "when",
    [lambda event: True, lambda event: event.kind == "attempt", lambda event: event.kind == "stage-start"],
    ids=["first-event", "first-attempt-event", "first-stage-start"],
)
def test_a_raising_observer_is_dropped_for_the_rest_of_the_run(
    when: Callable[[Any], bool], tmp_path: Path, bench: Path
) -> None:
    recipe = write_recipe(tmp_path, "p48-hook")
    raiser = Raiser(when)
    run(recipe, tmp_path / "runs", bench, make_registry(fake_toolchain(fail_first=True)), raiser)
    assert raiser.raised_at is not None, "the observer was never given the event it raises at"
    assert raiser.calls == raiser.raised_at, "an observer that raised was called again"


def test_stages_get_the_observer_through_their_run_context(tmp_path: Path, bench: Path) -> None:
    contexts: list[RunContext] = []

    class Capturing(GenerateStage):
        """The real generate stage, keeping the RunContext it is built on."""

        def __init__(self, *, context: RunContext) -> None:
            """Keep the context and build the real stage."""
            contexts.append(context)
            super().__init__(context=context)

    recipe = write_recipe(tmp_path, "p48-hook")
    run(recipe, tmp_path / "runs", bench, make_registry(fake_toolchain(), {"generate": Capturing}), Recorder())
    assert len(contexts) == 2, "one RunContext per trial"
    assert all(context.observer is not None for context in contexts)


# ---------------------------------------------------------------------------
# What the events carry


SCENARIOS = {
    "clean": ({}, {}),
    "one-correction": ({"fail_first": True}, {}),
    "correction-cap": ({"fail_always": True}, {"loop": {"max_corrections": 2}}),
}


@pytest.mark.parametrize(("toolchain", "changes"), list(SCENARIOS.values()), ids=list(SCENARIOS))
def test_each_trial_is_framed_by_trial_start_and_trial_end_carrying_the_written_trial(
    toolchain: dict[str, bool], changes: dict[str, Any], tmp_path: Path, bench: Path
) -> None:
    recorder = Recorder()
    run_dir = run(write_recipe(tmp_path, "p48-hook", **changes), tmp_path / "runs", bench,
                  make_registry(fake_toolchain(**toolchain)), recorder)
    assert all(event.kind in KINDS for event in recorder.events)
    assert all(isinstance(event.trial, Trial) for event in recorder.events)
    per_trial = trials_of(recorder.events)
    assert len(per_trial) == 2
    for number, events in enumerate(per_trial, start=1):
        kinds = [event.kind for event in events]
        assert kinds[0] == "trial-start" and kinds[-1] == "trial-end", kinds
        assert kinds.count("trial-start") == 1 and kinds.count("trial-end") == 1, kinds
        trial_id = events[-1].trial.trial_id
        assert trial_id.endswith(f"/run{number:02d}")
        assert all(event.trial.trial_id == trial_id for event in events)
        assert events[-1].trial == written_trial(run_dir, trial_id), "trial-end carries the trial as written"


def test_trial_start_carries_the_run_the_place_and_the_source(tmp_path: Path, bench: Path) -> None:
    recorder = Recorder()
    run(write_recipe(tmp_path, "p48-hook"), tmp_path / "runs", bench, make_registry(fake_toolchain()), recorder)
    starts = [event for event in recorder.events if event.kind == "trial-start"]
    assert [(event.run_id, event.number, event.count) for event in starts] == [(RUN_ID, 1, 2), (RUN_ID, 2, 2)]
    assert all(event.source == {"main.cpp": FAKE_OMP_SOURCE} for event in starts)
    for event in starts:
        assert event.trial.attempts == [] and event.trial.requests == [], "a snapshot never changes afterwards"


@pytest.mark.parametrize(("toolchain", "changes"), list(SCENARIOS.values()), ids=list(SCENARIOS))
def test_stage_start_names_each_stage_the_runner_runs_in_order(
    toolchain: dict[str, bool], changes: dict[str, Any], tmp_path: Path, bench: Path
) -> None:
    recorder = Recorder()
    run(write_recipe(tmp_path, "p48-hook", **changes), tmp_path / "runs", bench,
        make_registry(fake_toolchain(**toolchain)), recorder)
    for events in trials_of(recorder.events):
        started = [event.stage for event in events if event.kind == "stage-start"]
        assert started == ["generate", "compile_loop"]


@pytest.mark.parametrize(("toolchain", "changes"), list(SCENARIOS.values()), ids=list(SCENARIOS))
def test_request_sent_comes_once_per_model_call_before_its_reply_is_recorded(
    toolchain: dict[str, bool], changes: dict[str, Any], tmp_path: Path, bench: Path
) -> None:
    recorder = Recorder()
    run(write_recipe(tmp_path, "p48-hook", **changes), tmp_path / "runs", bench,
        make_registry(fake_toolchain(**toolchain)), recorder)
    for events in trials_of(recorder.events):
        final = events[-1].trial
        sent = [event for event in events if event.kind == "request-sent"]
        assert final.requests is not None
        assert len(sent) == len(final.requests), "one request-sent event per model call"
        assert [event.stage for event in sent] == [request.stage for request in final.requests]
        for count, event in enumerate(sent):
            assert len(event.trial.requests or []) == count, "the request is sent before its reply is recorded"
        kinds = [event.kind for event in events]
        assert kinds.index("request-sent") < kinds.index("attempt"), "the translation is asked before it arrives"


@pytest.mark.parametrize(("toolchain", "changes"), list(SCENARIOS.values()), ids=list(SCENARIOS))
def test_attempt_events_show_each_attempt_appended_and_then_as_built(
    toolchain: dict[str, bool], changes: dict[str, Any], tmp_path: Path, bench: Path
) -> None:
    recorder = Recorder()
    run(write_recipe(tmp_path, "p48-hook", **changes), tmp_path / "runs", bench,
        make_registry(fake_toolchain(**toolchain)), recorder)
    for events in trials_of(recorder.events):
        final = events[-1].trial
        shown = [(position, event) for position, event in enumerate(events) if event.kind == "attempt"]
        assert shown, "no attempt event"
        counts = [len(event.trial.attempts) for _, event in shown]
        assert counts == sorted(counts) and set(counts) == set(range(1, len(final.attempts) + 1)), counts
        assert shown[-1][1].trial.attempts == final.attempts, "the last attempt event shows every attempt as it ends"
        sent = [position for position, event in enumerate(events) if event.kind == "request-sent"]
        for index, attempt in enumerate(final.attempts):
            appended = next(event for _, event in shown if len(event.trial.attempts) > index)
            assert appended.trial.attempts[index].stage_reached == "S1", "appended before it is built"
            built = [position for position, event in shown
                     if len(event.trial.attempts) > index and event.trial.attempts[index] == attempt]
            assert built, f"attempt {index} is never shown as it ends"
            if index + 1 < len(final.attempts):
                asking = sent[final.requests.index(next(r for r in final.requests if r.attempt_index == index + 1))]
                assert built[0] < asking, f"attempt {index}'s build is shown before its correction is asked"


def test_a_trial_ended_by_the_baseline_sends_no_model_events(tmp_path: Path) -> None:
    bench = write_bench(tmp_path / "bench", cuda_source=BROKEN_CUDA_SOURCE)
    recipe = write_recipe(
        tmp_path, "p48-hook", stages=["baseline", "generate", "compile_loop"], fixes={"baseline_both": False},
        trials={"n": 1},
    )
    recorder = Recorder()
    run(recipe, tmp_path / "runs", bench, make_registry(fake_toolchain()), recorder)
    kinds = [event.kind for event in recorder.events]
    assert kinds == ["trial-start", "stage-start", "trial-end"], kinds
    assert recorder.events[1].stage == "baseline"
    ended = recorder.events[-1].trial
    assert ended.final.end_reason is not None and ended.final.end_reason.code == "baseline-compile"


# ---------------------------------------------------------------------------
# The notice a dropped observer leaves, and a stderr that cannot take it


class BrokenStderr(io.StringIO):
    """A stderr whose every write and flush raises BrokenPipeError, as a pipe whose reader has gone does."""

    def write(self, text: str) -> int:
        """Refuse the text."""
        raise BrokenPipeError(32, "SYNTHETIC broken pipe")

    def flush(self) -> None:
        """Refuse the flush."""
        raise BrokenPipeError(32, "SYNTHETIC broken pipe")


def closed_stderr() -> io.StringIO:
    """Return a closed stream: every write raises ValueError."""
    stream = io.StringIO()
    stream.close()
    return stream


STDERRS: dict[str, Callable[[], Any]] = {"broken-pipe": BrokenStderr, "closed": closed_stderr, "none": lambda: None}


@pytest.mark.parametrize("kind", list(STDERRS), ids=list(STDERRS))
def test_an_observer_whose_failure_also_breaks_stderr_leaves_the_same_tree(
    kind: str, tmp_path: Path, bench: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = write_recipe(tmp_path, "p48-hook")
    registry = make_registry(fake_toolchain(fail_first=True))
    plain = run(recipe, tmp_path / "plain", bench, registry)
    calls: list[str] = []

    def breaks_stderr(event: Any) -> None:
        """Break stderr, then fail as a write to it would."""
        calls.append(event.kind)
        monkeypatch.setattr(sys, "stderr", STDERRS[kind]())
        raise BrokenPipeError(32, "SYNTHETIC broken pipe")

    observed = run(recipe, tmp_path / "observed", bench, registry, breaks_stderr)
    assert calls == ["trial-start"], "the observer that raised was called again"
    assert_same_tree(observed, plain)
    manifest = json.loads((observed / "provenance.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"


def test_a_dropped_observer_leaves_one_stderr_line_naming_the_exception_class(
    tmp_path: Path, bench: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raiser = Raiser(lambda event: event.kind == "attempt")
    run(write_recipe(tmp_path, "p48-hook"), tmp_path / "runs", bench, make_registry(fake_toolchain()), raiser)
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 1, lines
    assert "RuntimeError" in lines[0] and "SYNTHETIC" not in lines[0], lines[0]


class DescriptorStderr(BrokenStderr):
    """A BrokenStderr that reports a chosen file descriptor, as the process's stderr reports 2."""

    def __init__(self, descriptor: int) -> None:
        """Report `descriptor` from fileno()."""
        super().__init__()
        self.descriptor = descriptor

    def fileno(self) -> int:
        """Return the chosen descriptor."""
        return self.descriptor


def test_a_notice_that_stderr_refuses_points_stderr_at_the_null_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = progress()
    muted, kept = tmp_path / "muted.txt", tmp_path / "kept.txt"
    with muted.open("wb") as first, kept.open("wb") as second:
        monkeypatch.setattr(sys, "stderr", DescriptorStderr(first.fileno()))
        module.stderr_notice("SYNTHETIC notice")
        module.mute_stderr(DescriptorStderr(second.fileno()))
        for target in (first, second):
            os.write(target.fileno(), b"SYNTHETIC bytes\n")
    assert muted.read_bytes() == b"", "a notice that stderr refused left stderr's descriptor as it was"
    assert kept.read_bytes() == b"SYNTHETIC bytes\n", "a stream other than sys.stderr was pointed elsewhere"
    monkeypatch.setattr(sys, "stderr", DescriptorStderr(-1))
    module.stderr_notice("SYNTHETIC notice")  # dup2 onto a bad descriptor fails; the notice raises nothing
